"""Differentiable implementation of the dense legacy SPEED value edit.

The module deliberately keeps model loading and prompt encoding outside its
interface.  Callers provide contextual embeddings and receive graph-connected
parameter overrides suitable for ``torch.func.functional_call``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

SPEED_CHECKPOINT_FORMAT = "speed-differentiable-anchor-v1"


@dataclass(frozen=True)
class DifferentiableLegacyEditConfig:
    residual_scale: float = 1.0
    retain_scale: float = 1.0
    threshold: float = 0.1
    lamb: float = 0.0
    chunk_size: int = 128
    seed: int = 0
    use_k2: bool = True

    def validate(self) -> None:
        if not isinstance(self.use_k2, bool):
            raise TypeError(f"use_k2 must be a boolean, got {type(self.use_k2).__name__}")
        values = {
            "residual_scale": self.residual_scale,
            "retain_scale": self.retain_scale,
            "threshold": self.threshold,
            "lamb": self.lamb,
        }
        for name, value in values.items():
            if not torch.isfinite(torch.tensor(float(value))):
                raise ValueError(f"{name} must be finite, got {value}")
        if self.residual_scale <= 0:
            raise ValueError("residual_scale must be positive")
        if self.retain_scale <= 0:
            raise ValueError("retain_scale must be positive")
        if self.lamb < 0:
            raise ValueError("lamb must be nonnegative")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")


def _as_embedding_tensor(
    value: torch.Tensor | Sequence[torch.Tensor],
    name: str,
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value
    else:
        values = list(value)
        if not values:
            raise ValueError(f"{name} must be non-empty")
        tensor = torch.stack(values)
    if tensor.ndim != 3 or tensor.shape[1] != 1:
        raise ValueError(f"{name} must have shape [N, 1, d], got {tuple(tensor.shape)}")
    if tensor.shape[0] == 0 or tensor.shape[2] == 0:
        raise ValueError(f"{name} must be non-empty")
    if not torch.is_floating_point(tensor):
        raise TypeError(f"{name} must be floating point")
    if not torch.isfinite(tensor).all().item():
        raise ValueError(f"{name} contains non-finite values")
    return tensor


def build_legacy_target_anchor_statistics(
    target_embeddings: torch.Tensor | Sequence[torch.Tensor],
    anchor_embeddings: torch.Tensor | Sequence[torch.Tensor],
    residual_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the exact dense statistics used by legacy SPEED."""

    targets = _as_embedding_tensor(target_embeddings, "target_embeddings")
    anchors = _as_embedding_tensor(anchor_embeddings, "anchor_embeddings")
    if anchors.shape != targets.shape:
        raise ValueError(
            "target_embeddings and anchor_embeddings must have matching shapes, "
            f"got {tuple(targets.shape)} and {tuple(anchors.shape)}"
        )
    if anchors.device != targets.device or anchors.dtype != targets.dtype:
        raise ValueError("target_embeddings and anchor_embeddings must share device and dtype")
    if not torch.isfinite(torch.tensor(float(residual_scale))).item():
        raise ValueError("residual_scale must be finite")

    residuals = residual_scale * (anchors - targets)
    sum_target_target = torch.stack(
        [target.T @ target for target in targets]
    ).mean(0)
    target_anchor_delta = torch.stack(
        [residual.T @ target for residual, target in zip(residuals, targets)]
    ).mean(0)
    return sum_target_target, target_anchor_delta


def build_retain_covariance(
    retain_embeddings: torch.Tensor | Sequence[torch.Tensor],
    chunk_size: int = 128,
    permutation: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build the legacy chunk-reduced retain covariance."""

    retains = _as_embedding_tensor(retain_embeddings, "retain_embeddings")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if permutation is not None:
        if permutation.ndim != 1 or permutation.numel() != retains.shape[0]:
            raise ValueError("retain permutation has the wrong shape")
        expected = torch.arange(retains.shape[0], device=permutation.device)
        if not torch.equal(torch.sort(permutation).values, expected):
            raise ValueError("retain permutation must contain every row exactly once")
        retains = retains[permutation.to(retains.device)]

    sums = []
    count = 0
    for start in range(0, retains.shape[0], chunk_size):
        chunk = retains[start : start + chunk_size]
        sums.append((chunk.transpose(1, 2) @ chunk).sum(0))
        count += chunk.shape[0]
    covariance = torch.stack(sums).sum(0) / count
    if not torch.isfinite(covariance).all().item():
        raise ValueError("retain covariance contains non-finite values")
    return covariance


def build_retain_projector(
    retain_covariance: torch.Tensor,
    threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if retain_covariance.ndim != 2 or retain_covariance.shape[0] != retain_covariance.shape[1]:
        raise ValueError("retain_covariance must be square")
    if not torch.isfinite(retain_covariance).all().item():
        raise ValueError("retain_covariance contains non-finite values")
    if not torch.isfinite(torch.tensor(float(threshold))).item():
        raise ValueError("threshold must be finite")
    left, singular_values, _ = torch.svd(retain_covariance)
    low_mask = singular_values < threshold
    projector = left[:, low_mask] @ left[:, low_mask].T
    return projector, singular_values


def build_k2(
    null_hidden_states: torch.Tensor,
    seed: int = 0,
) -> torch.Tensor:
    """Build the BOS plus three-center null-prompt constraint matrix."""

    hidden = null_hidden_states
    if hidden.ndim == 3 and hidden.shape[0] == 1:
        hidden = hidden[0]
    if hidden.ndim != 2 or hidden.shape[0] < 4:
        raise ValueError(
            "null_hidden_states must have shape [sequence_length, d] with at least four rows"
        )
    if not torch.isfinite(hidden).all().item():
        raise ValueError("null_hidden_states contains non-finite values")
    try:
        from kmeans_pytorch import kmeans
    except ImportError as error:
        raise RuntimeError("kmeans_pytorch is required to construct K2") from error

    cuda_devices = []
    if hidden.device.type == "cuda":
        cuda_devices = [hidden.device.index or torch.cuda.current_device()]
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(seed)
        if hidden.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        _, centers = kmeans(
            X=hidden[1:],
            num_clusters=3,
            distance="euclidean",
            device=str(hidden.device),
        )
    return torch.cat([hidden[[0]], centers.to(hidden)], dim=0).T


def select_value_parameter_names(module: torch.nn.Module) -> list[str]:
    names = [
        name
        for name, _ in module.named_parameters()
        if name.endswith("attn2.to_v.weight")
    ]
    if not names:
        raise ValueError("No attn2.to_v.weight parameters were found")
    return names


def _tensor_fingerprint(tensor: torch.Tensor) -> str:
    value = tensor.detach().to(device="cpu").contiguous()
    payload = value.numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


class DifferentiableLegacyEditState:
    """Frozen SPEED geometry with a differentiable anchor-to-weight interface."""

    def __init__(
        self,
        *,
        target_embeddings: torch.Tensor,
        sum_target_target: torch.Tensor,
        retain_covariance: torch.Tensor,
        retain_projector: torch.Tensor,
        retain_singular_values: torch.Tensor,
        k2: torch.Tensor | None,
        matrix_m: torch.Tensor,
        inner_inverse: torch.Tensor | None,
        base_weights: Mapping[str, torch.Tensor],
        config: DifferentiableLegacyEditConfig,
    ) -> None:
        self.target_embeddings = target_embeddings.detach().clone()
        self.sum_target_target = sum_target_target.detach().clone()
        self.retain_covariance = retain_covariance.detach().clone()
        self.retain_projector = retain_projector.detach().clone()
        self.retain_singular_values = retain_singular_values.detach().clone()
        self.k2 = k2.detach().clone() if k2 is not None else None
        self.matrix_m = matrix_m.detach().clone()
        self.inner_inverse = (
            inner_inverse.detach().clone() if inner_inverse is not None else None
        )
        self.base_weights = {
            name: weight.detach().clone() for name, weight in base_weights.items()
        }
        self.config = config

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(self.base_weights)

    @property
    def projector_rank(self) -> int:
        return int((self.retain_singular_values < self.config.threshold).sum().item())

    def effective_parameters(
        self,
        anchor_embeddings: torch.Tensor | Sequence[torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
        anchors = _as_embedding_tensor(anchor_embeddings, "anchor_embeddings")
        targets = self.target_embeddings
        if anchors.shape != targets.shape:
            raise ValueError(
                f"anchor_embeddings must have shape {tuple(targets.shape)}, got {tuple(anchors.shape)}"
            )
        if anchors.device != targets.device or anchors.dtype != targets.dtype:
            raise ValueError("anchor_embeddings must share target device and dtype")

        _, target_anchor_delta = build_legacy_target_anchor_statistics(
            targets,
            anchors,
            residual_scale=self.config.residual_scale,
        )
        dimension = targets.shape[-1]
        if self.config.use_k2:
            if self.k2 is None or self.inner_inverse is None:
                raise RuntimeError("k2 and inner_inverse are required when use_k2 is True")
            identity = torch.eye(dimension, device=targets.device, dtype=targets.dtype)
            correction = (
                identity
                - self.matrix_m
                @ self.k2
                @ self.inner_inverse
                @ self.k2.T
                @ self.retain_projector
            )
        else:
            correction = None

        overrides: dict[str, torch.Tensor] = {}
        update_norms: dict[str, float] = {}
        for name, base_weight in self.base_weights.items():
            if self.config.use_k2:
                delta_weight = (
                    base_weight
                    @ target_anchor_delta
                    @ self.retain_projector
                    @ correction
                    @ self.matrix_m
                )
            else:
                delta_weight = (
                    base_weight
                    @ target_anchor_delta
                    @ self.retain_projector
                    @ self.matrix_m
                )
            effective_weight = base_weight + delta_weight
            if not torch.isfinite(effective_weight).all().item():
                raise FloatingPointError(f"Effective weight is non-finite for {name}")
            overrides[name] = effective_weight
            update_norms[name] = float(
                torch.linalg.vector_norm(delta_weight.detach().float()).item()
            )

        residuals = anchors - targets
        diagnostics: dict[str, object] = {
            "projector_rank": self.projector_rank,
            "residual_norms": torch.linalg.vector_norm(
                residuals.detach().float().flatten(1), dim=1
            ).cpu().tolist(),
            "scaled_residual_norms": torch.linalg.vector_norm(
                (self.config.residual_scale * residuals).detach().float().flatten(1),
                dim=1,
            ).cpu().tolist(),
            "target_anchor_delta_norm": float(
                torch.linalg.vector_norm(target_anchor_delta.detach().float()).item()
            ),
            "layer_update_norms": update_norms,
        }
        return overrides, diagnostics

    @torch.no_grad()
    def materialize(
        self,
        anchor_embeddings: torch.Tensor | Sequence[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        overrides, _ = self.effective_parameters(anchor_embeddings)
        return {
            name: value.detach().to(device="cpu").contiguous()
            for name, value in overrides.items()
        }

    def geometry_metadata(self) -> dict[str, object]:
        return {
            "use_k2": self.config.use_k2,
            "projector_rank": self.projector_rank,
            "target_count": int(self.target_embeddings.shape[0]),
            "embedding_dimension": int(self.target_embeddings.shape[-1]),
            "parameter_names": list(self.parameter_names),
            "target_embeddings_sha256": _tensor_fingerprint(self.target_embeddings),
            "sum_target_target_sha256": _tensor_fingerprint(self.sum_target_target),
            "retain_covariance_sha256": _tensor_fingerprint(self.retain_covariance),
            "retain_projector_sha256": _tensor_fingerprint(self.retain_projector),
            "k2_sha256": _tensor_fingerprint(self.k2) if self.k2 is not None else None,
        }


def _inverse(matrix: torch.Tensor, label: str) -> torch.Tensor:
    if not torch.isfinite(matrix).all().item():
        raise ValueError(f"{label} contains non-finite values")
    try:
        inverse = torch.linalg.inv(matrix)
    except RuntimeError as error:
        raise ValueError(f"{label} is singular and cannot be inverted") from error
    if not torch.isfinite(inverse).all().item():
        raise ValueError(f"Inverse of {label} contains non-finite values")
    return inverse


def prepare_differentiable_legacy_edit(
    base_unet: torch.nn.Module,
    target_embeddings: torch.Tensor | Sequence[torch.Tensor],
    retain_embeddings: torch.Tensor | Sequence[torch.Tensor],
    null_hidden_states: torch.Tensor | None = None,
    config: DifferentiableLegacyEditConfig = DifferentiableLegacyEditConfig(),
    *,
    prepared_k2: torch.Tensor | None = None,
    retain_permutation: torch.Tensor | None = None,
) -> DifferentiableLegacyEditState:
    """Prepare anchor-independent SPEED geometry and frozen value weights."""

    config.validate()
    if not hasattr(torch, "func") or not hasattr(torch.func, "functional_call"):
        raise RuntimeError("This PyTorch build does not provide torch.func.functional_call")

    targets = _as_embedding_tensor(target_embeddings, "target_embeddings")
    retains = _as_embedding_tensor(retain_embeddings, "retain_embeddings")
    if retains.shape[-1] != targets.shape[-1]:
        raise ValueError("retain and target embedding dimensions must match")
    if retains.device != targets.device or retains.dtype != targets.dtype:
        raise ValueError("retain and target embeddings must share device and dtype")

    named_parameters = dict(base_unet.named_parameters())
    selected_names = select_value_parameter_names(base_unet)
    base_weights = {name: named_parameters[name] for name in selected_names}
    for name, weight in base_weights.items():
        if weight.ndim != 2 or weight.shape[1] != targets.shape[-1]:
            raise ValueError(
                f"{name} must be a matrix with input width {targets.shape[-1]}, "
                f"got {tuple(weight.shape)}"
            )
        if weight.device != targets.device or weight.dtype != targets.dtype:
            raise ValueError(f"{name} must share target device and dtype")
        if not torch.isfinite(weight).all().item():
            raise ValueError(f"{name} contains non-finite values")

    dimension = targets.shape[-1]
    sum_target_target, _ = build_legacy_target_anchor_statistics(targets, targets)
    retain_covariance = build_retain_covariance(
        retains,
        chunk_size=config.chunk_size,
        permutation=retain_permutation,
    )
    retain_projector, singular_values = build_retain_projector(
        retain_covariance,
        config.threshold,
    )

    identity = torch.eye(dimension, device=targets.device, dtype=targets.dtype)
    matrix_m = _inverse(
        sum_target_target @ retain_projector + config.retain_scale * identity,
        "C @ P + retain_scale * I",
    )

    if config.use_k2:
        if prepared_k2 is None:
            if null_hidden_states is None:
                raise ValueError("null_hidden_states is required when use_k2 is True")
            k2 = build_k2(null_hidden_states, seed=config.seed)
        else:
            k2 = prepared_k2
        if k2.ndim != 2 or k2.shape != (dimension, 4):
            raise ValueError(f"K2 must have shape [{dimension}, 4], got {tuple(k2.shape)}")
        if k2.device != targets.device or k2.dtype != targets.dtype:
            raise ValueError("K2 must share target device and dtype")
        if not torch.isfinite(k2).all().item():
            raise ValueError("K2 contains non-finite values")

        identity2 = torch.eye(4, device=targets.device, dtype=targets.dtype)
        inner_inverse = _inverse(
            k2.T @ retain_projector @ matrix_m @ k2 + config.lamb * identity2,
            "K2.T @ P @ M @ K2 + lamb * I2",
        )
    else:
        k2 = None
        inner_inverse = None

    return DifferentiableLegacyEditState(
        target_embeddings=targets,
        sum_target_target=sum_target_target,
        retain_covariance=retain_covariance,
        retain_projector=retain_projector,
        retain_singular_values=singular_values,
        k2=k2,
        matrix_m=matrix_m,
        inner_inverse=inner_inverse,
        base_weights=base_weights,
        config=config,
    )
