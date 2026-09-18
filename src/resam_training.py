"""ReSAM: Retain-Guided Sparse Anchor Mixture training and numerical primitives."""

from __future__ import annotations

import csv
import hashlib
import math
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F

from src.closed_form_anchor_training import (
    DiffusionState,
    paired_predictions,
    predicted_noise_cosine,
)
from src.differentiable_legacy_edit import DifferentiableLegacyEditState


def load_candidate_concepts(
    path: str | os.PathLike[str],
) -> tuple[list[str], str]:
    """Load candidate bank from CSV or text file in order, returning concepts and sha256 hash.

    Rejects empty banks, blank concepts, and duplicates.
    """
    path_obj = Path(path)
    if not path_obj.is_file():
        raise FileNotFoundError(f"Candidate concepts file not found: {path}")

    hasher = hashlib.sha256()
    with open(path_obj, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            hasher.update(chunk)
    file_hash = hasher.hexdigest()

    candidates: list[str] = []
    seen: set[str] = set()

    if str(path_obj).endswith(".csv"):
        with open(path_obj, newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            col = None
            for candidate_col in ("concept", "prompt"):
                if reader.fieldnames and candidate_col in reader.fieldnames:
                    col = candidate_col
                    break
            if col is None:
                raise ValueError(
                    f"Candidate CSV must contain a 'concept' or 'prompt' column: {path}"
                )
            for row_num, row in enumerate(reader, start=2):
                val = (row.get(col) or "").strip()
                if not val:
                    raise ValueError(
                        f"Candidate CSV contains a blank concept at row {row_num}: {path}"
                    )
                if val in seen:
                    raise ValueError(
                        f"Candidate bank contains duplicate concept {val!r} at row {row_num}: {path}"
                    )
                seen.add(val)
                candidates.append(val)
    else:
        with open(path_obj, encoding="utf-8") as txt_file:
            for line_num, line in enumerate(txt_file, start=1):
                val = line.strip()
                if not val:
                    raise ValueError(
                        f"Candidate file contains a blank line at line {line_num}: {path}"
                    )
                if val in seen:
                    raise ValueError(
                        f"Candidate bank contains duplicate concept {val!r} at line {line_num}: {path}"
                    )
                seen.add(val)
                candidates.append(val)

    if not candidates:
        raise ValueError(f"Candidate bank is empty: {path}")

    return candidates, file_hash


class SparseAnchorMixture(torch.nn.Module):
    """Learn a sparse top-k mixture over a frozen bank of candidate concept embeddings."""

    def __init__(
        self,
        candidate_embeddings: torch.Tensor,
        target_embeddings: torch.Tensor,
        k: int,
        temperature: float = 1.0,
        *,
        use_ste: bool = True,
        candidate_names: Sequence[str] | None = None,
        initial_scores: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if not torch.is_floating_point(candidate_embeddings):
            raise TypeError("candidate_embeddings must be floating point")
        if not torch.is_floating_point(target_embeddings):
            raise TypeError("target_embeddings must be floating point")
        if candidate_embeddings.ndim == 2:
            candidate_embeddings = candidate_embeddings.unsqueeze(1)
        if candidate_embeddings.ndim != 3 or candidate_embeddings.shape[1] != 1:
            raise ValueError("candidate_embeddings must have shape [V, 1, d] or [V, d]")
        if target_embeddings.ndim != 3 or target_embeddings.shape[:2] != (1, 1):
            raise ValueError("target_embeddings must have shape [1, 1, d]")
        if candidate_embeddings.shape[2] != target_embeddings.shape[2]:
            raise ValueError("candidate_embeddings and target_embeddings width must match")
        if (
            candidate_embeddings.device != target_embeddings.device
            or candidate_embeddings.dtype != target_embeddings.dtype
        ):
            raise ValueError("candidate_embeddings and target_embeddings must share device and dtype")
        if not torch.isfinite(candidate_embeddings).all().item():
            raise ValueError("candidate_embeddings contains non-finite values")
        if not torch.isfinite(target_embeddings).all().item():
            raise ValueError("target_embeddings contains non-finite values")

        v_count = candidate_embeddings.shape[0]
        if not isinstance(k, int) or isinstance(k, bool):
            raise TypeError("k must be an integer")
        if not 1 <= k <= v_count:
            raise ValueError(f"k must be between 1 and bank size ({v_count}), got {k}")
        if not isinstance(temperature, (int, float)) or isinstance(temperature, bool):
            raise TypeError("temperature must be a positive float")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be positive and finite")

        if candidate_names is not None:
            if len(candidate_names) != v_count:
                raise ValueError("candidate_names length must match candidate bank size")
            self.candidate_names = list(candidate_names)
        else:
            self.candidate_names = [f"candidate_{i}" for i in range(v_count)]

        self.k = k
        self.temperature = float(temperature)
        self.use_ste = bool(use_ste)

        self.register_buffer("candidate_embeddings", candidate_embeddings.detach().clone())
        self.register_buffer("target_embeddings", target_embeddings.detach().clone())

        if initial_scores is not None:
            if initial_scores.shape != (v_count,):
                raise ValueError(f"initial_scores must have shape [{v_count}]")
            if not torch.isfinite(initial_scores).all().item():
                raise ValueError("initial_scores contains non-finite values")
            scores = initial_scores.detach().clone().float().to(device=candidate_embeddings.device)
        else:
            scores = torch.zeros(v_count, dtype=torch.float32, device=candidate_embeddings.device)

        self.raw_scores = torch.nn.Parameter(scores)
        self._last_selected_indices: tuple[int, ...] | None = None
        self._selection_changes = 0

    def compute_weights(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute sparse mixture weights with optional straight-through estimation.

        Returns:
            weights: Effective gradient-bearing weights [V] summing to 1.
            hard_weights: Normalized sparse weights [V] (zeros outside top-k).
            topk_indices: Indices of the selected top-k candidates in bank order for ties.
        """
        if not torch.isfinite(self.raw_scores).all().item():
            raise FloatingPointError("Scores contain non-finite values")

        soft = torch.softmax(self.raw_scores / self.temperature, dim=0)

        # Deterministic bank-order tie rule using stable argsort descending
        sorted_indices = torch.argsort(self.raw_scores, descending=True, stable=True)
        topk_indices = sorted_indices[: self.k]

        hard = torch.zeros_like(soft)
        hard[topk_indices] = torch.softmax(self.raw_scores[topk_indices] / self.temperature, dim=0)

        if self.use_ste:
            weights = hard.detach() - soft.detach() + soft
        else:
            weights = hard

        return weights, hard, topk_indices

    def forward(self) -> torch.Tensor:
        """Synthesize anchor embedding a(s) of shape [1, 1, d]."""
        weights, _, topk_indices = self.compute_weights()

        current_indices = tuple(sorted(topk_indices.cpu().tolist()))
        if self._last_selected_indices is not None and current_indices != self._last_selected_indices:
            self._selection_changes += 1
        self._last_selected_indices = current_indices

        # a(s) = sum_j w_j * e_j
        weighted = (
            weights.to(dtype=self.candidate_embeddings.dtype).view(-1, 1, 1)
            * self.candidate_embeddings
        ).sum(dim=0, keepdim=True)
        return weighted

    def residuals(self) -> torch.Tensor:
        """Compute single-target residual [1, 1, d] = anchor - target."""
        return self.forward() - self.target_embeddings

    def diagnostics(self) -> dict[str, object]:
        """Expose detached selection, weights, norms, and distance statistics."""
        with torch.no_grad():
            weights, hard, topk_indices = self.compute_weights()
            anchor = self.forward()
            residual = self.residuals()

            anchor_norm = float(torch.linalg.vector_norm(anchor.float().flatten()).item())
            residual_norm = float(torch.linalg.vector_norm(residual.float().flatten()).item())
            anchor_target_dist = float(
                torch.linalg.vector_norm((anchor - self.target_embeddings).float().flatten()).item()
            )

            selected_idx = topk_indices.cpu().tolist()
            selected_names = [self.candidate_names[i] for i in selected_idx]
            selected_weights = hard[topk_indices].cpu().tolist()

            # Weight entropy across active candidates
            active = hard[topk_indices].clamp_min(1e-12)
            entropy = float(-(active * torch.log(active)).sum().item())

            return {
                "selected_indices": selected_idx,
                "selected_names": selected_names,
                "selected_weights": selected_weights,
                "all_scores": self.raw_scores.detach().cpu().tolist(),
                "anchor_norm": anchor_norm,
                "residual_norm": residual_norm,
                "anchor_target_distance": anchor_target_dist,
                "weight_entropy": entropy,
                "selection_changes": self._selection_changes,
            }


def retain_noise_mse(
    base_prediction: torch.Tensor,
    edited_prediction: torch.Tensor,
) -> torch.Tensor:
    """Mean squared error between teacher and edited predicted noise."""
    if base_prediction.shape != edited_prediction.shape:
        raise ValueError("Base and edited predictions must share shape")
    return F.mse_loss(edited_prediction.float(), base_prediction.float())


@dataclass(frozen=True)
class ReSAMTrainingConfig:
    steps: int
    learning_rate: float
    use_null_retain_loss: bool = False

    def validate(self) -> None:
        if not isinstance(self.steps, int) or isinstance(self.steps, bool) or self.steps <= 0:
            raise ValueError("steps must be a positive integer")
        if (
            isinstance(self.learning_rate, bool)
            or not isinstance(self.learning_rate, (int, float))
            or not math.isfinite(self.learning_rate)
            or self.learning_rate <= 0
        ):
            raise ValueError("learning_rate must be positive and finite")
        if not isinstance(self.use_null_retain_loss, bool):
            raise ValueError("use_null_retain_loss must be a boolean")


@dataclass
class ReSAMTrainingResult:
    final_step: int
    final_loss: float
    history: list[dict[str, object]]


def optimize_resam(
    base_unet: torch.nn.Module,
    edit_state: DifferentiableLegacyEditState,
    mixture: SparseAnchorMixture,
    training_state_factory: Callable[[], DiffusionState],
    config: ReSAMTrainingConfig,
    *,
    metrics_callback: Callable[[dict[str, object]], None] | None = None,
) -> ReSAMTrainingResult:
    """Optimize candidate scores to minimize retain (and optional null) predicted-noise MSE."""

    config.validate()
    if any(p.requires_grad for p in base_unet.parameters()):
        raise ValueError("base_unet parameters must be frozen before ReSAM training")
    if base_unet.training:
        raise ValueError("base_unet must be in eval mode before ReSAM training")

    optimizer = torch.optim.Adam(mixture.parameters(), lr=config.learning_rate)
    history: list[dict[str, object]] = []

    last_loss = 0.0
    for step in range(1, config.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        anchor = mixture()
        overrides, edit_diagnostics = edit_state.effective_parameters(anchor)

        diffusion_state = training_state_factory()
        base_pred, edited_pred = paired_predictions(base_unet, overrides, diffusion_state)
        retain_loss = retain_noise_mse(base_pred, edited_pred)

        if config.use_null_retain_loss:
            if diffusion_state.null_hidden_states is None:
                raise ValueError(
                    "null_hidden_states is required when use_null_retain_loss is True"
                )
            null_base_pred, null_edited_pred = paired_predictions(
                base_unet,
                overrides,
                diffusion_state,
                hidden_states=diffusion_state.null_hidden_states,
            )
            null_loss = retain_noise_mse(null_base_pred, null_edited_pred)
            loss = 0.5 * (retain_loss + null_loss)
        else:
            null_loss = None
            loss = retain_loss

        if not torch.isfinite(loss).item():
            raise FloatingPointError(f"Non-finite retain loss at step {step}: {loss.item()}")

        loss.backward()

        squared_grad_norm = 0.0
        for name, param in mixture.named_parameters():
            if param.grad is None:
                raise RuntimeError(f"Parameter {name} has no gradient at step {step}")
            if not torch.isfinite(param.grad).all().item():
                raise FloatingPointError(f"Non-finite gradient in {name} at step {step}")
            squared_grad_norm += float(param.grad.detach().float().square().sum().item())

        grad_norm = math.sqrt(squared_grad_norm)
        step_diagnostics = mixture.diagnostics()
        optimizer.step()

        for name, param in mixture.named_parameters():
            if not torch.isfinite(param).all().item():
                raise FloatingPointError(f"Non-finite parameter {name} after step {step}")

        last_loss = float(loss.detach().item())
        step_record: dict[str, object] = {
            "step": step,
            "train_loss": last_loss,
            "retain_loss": float(retain_loss.detach().item()),
            "grad_norm": grad_norm,
            "prompt": diffusion_state.prompt,
            "prefix_index": diffusion_state.prefix_index,
            "state_seed": diffusion_state.seed,
            **step_diagnostics,
            **edit_diagnostics,
        }
        if null_loss is not None:
            step_record["null_loss"] = float(null_loss.detach().item())

        history.append(step_record)
        if metrics_callback is not None:
            metrics_callback(step_record)

    return ReSAMTrainingResult(
        final_step=config.steps,
        final_loss=last_loss,
        history=history,
    )
