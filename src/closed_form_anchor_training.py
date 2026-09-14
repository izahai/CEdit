"""Anchor-only optimization through differentiable SPEED weights."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from src.differentiable_legacy_edit import DifferentiableLegacyEditState


@dataclass(frozen=True)
class DiffusionState:
    model_input: torch.Tensor
    timestep: torch.Tensor
    target_hidden_states: torch.Tensor
    null_hidden_states: torch.Tensor | None = None
    prompt: str = ""
    prefix_index: int = 0
    seed: int = 0
    unet_kwargs: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class AnchorTrainingConfig:
    steps: int
    learning_rate: float
    validation_interval: int
    cosine_eps: float = 1e-8
    use_null_retain_loss: bool = False

    def validate(self) -> None:
        if self.steps <= 0:
            raise ValueError("steps must be positive")
        if self.learning_rate <= 0 or not math.isfinite(self.learning_rate):
            raise ValueError("learning_rate must be positive and finite")
        if self.validation_interval <= 0:
            raise ValueError("validation_interval must be positive")
        if self.cosine_eps <= 0 or not math.isfinite(self.cosine_eps):
            raise ValueError("cosine_eps must be positive and finite")
        if not isinstance(self.use_null_retain_loss, bool):
            raise ValueError("use_null_retain_loss must be a boolean")


@dataclass
class AnchorTrainingResult:
    best_step: int
    best_validation_cosine: float
    best_validation_loss: float
    best_validation_null_cosine: float | None
    best_raw_state: dict[str, torch.Tensor]
    history: list[dict[str, object]]


class BoundedAnchor(torch.nn.Module):
    """Train anchor direction and magnitude within each target-norm ball."""

    def __init__(
        self,
        target_embeddings: torch.Tensor,
        initial_anchor_embeddings: torch.Tensor,
        *,
        seed: int = 0,
        direction_eps: float = 1e-12,
        ratio_eps: float = 1e-4,
        zero_init_ratio: float = 1e-2,
    ) -> None:
        super().__init__()
        if target_embeddings.ndim != 3 or target_embeddings.shape[1] != 1:
            raise ValueError("target_embeddings must have shape [N, 1, d]")
        if initial_anchor_embeddings.shape != target_embeddings.shape:
            raise ValueError("initial anchors must match target embedding shape")
        if not torch.is_floating_point(target_embeddings):
            raise TypeError("target_embeddings must be floating point")
        if (
            initial_anchor_embeddings.device != target_embeddings.device
            or initial_anchor_embeddings.dtype != target_embeddings.dtype
        ):
            raise ValueError("initial anchors must share target device and dtype")
        if not math.isfinite(direction_eps) or direction_eps <= 0:
            raise ValueError("direction_eps must be positive and finite")
        if not 0 < ratio_eps < 0.5:
            raise ValueError("ratio_eps must be between zero and 0.5")
        if not 0 < zero_init_ratio < 1:
            raise ValueError("zero_init_ratio must be between zero and one")
        if not torch.isfinite(target_embeddings).all().item():
            raise ValueError("target_embeddings contains non-finite values")
        if not torch.isfinite(initial_anchor_embeddings).all().item():
            raise ValueError("initial_anchor_embeddings contains non-finite values")

        target = target_embeddings.detach().clone()
        initial_anchor = initial_anchor_embeddings.detach().clone()
        max_anchor_norm = torch.linalg.vector_norm(
            target.flatten(1), dim=1
        ).reshape(-1, 1, 1)
        if torch.any(max_anchor_norm <= direction_eps).item():
            raise ValueError("target_embeddings must have nonzero norm")
        flat = initial_anchor.flatten(1)
        norms = torch.linalg.vector_norm(flat, dim=1, keepdim=True)
        was_zero = norms <= direction_eps

        generator = torch.Generator(device="cpu").manual_seed(seed)
        random_directions = torch.randn(
            flat.shape,
            generator=generator,
            dtype=torch.float32,
            device="cpu",
        ).to(device=flat.device, dtype=flat.dtype)
        random_directions = random_directions / torch.linalg.vector_norm(
            random_directions, dim=1, keepdim=True
        ).clamp_min(direction_eps)

        directions = flat / norms.clamp_min(direction_eps)
        directions = torch.where(was_zero, random_directions, directions)
        ratios = norms / max_anchor_norm.flatten(1)
        ratios = torch.where(
            was_zero,
            torch.full_like(ratios, zero_init_ratio),
            ratios,
        )
        # Leave radial headroom when the text anchor starts outside the ball.
        clipped_ratio = min(0.95, 1.0 - ratio_eps)
        feasible_ratios = torch.where(
            ratios >= 1.0,
            torch.full_like(ratios, clipped_ratio),
            ratios.clamp(ratio_eps, 1.0 - ratio_eps),
        )
        feasible_anchor = (
            max_anchor_norm.flatten(1) * feasible_ratios * directions
        ).reshape_as(target)

        self.register_buffer("target_embeddings", target)
        self.register_buffer("original_initial_anchor", initial_anchor)
        self.register_buffer("feasible_initial_anchor", feasible_anchor)
        self.register_buffer("max_anchor_norm", max_anchor_norm)
        self.register_buffer("initial_was_zero", was_zero.flatten())
        self.register_buffer("initial_was_clipped", (ratios >= 1.0).flatten())
        self.direction_eps = float(direction_eps)
        self.raw_direction = torch.nn.Parameter(directions.reshape_as(target).clone())
        logits = torch.log(feasible_ratios) - torch.log1p(-feasible_ratios)
        self.raw_magnitude = torch.nn.Parameter(logits.reshape(-1, 1, 1).clone())

    def residuals(self) -> torch.Tensor:
        return self() - self.target_embeddings

    def forward(self) -> torch.Tensor:
        direction = self.raw_direction / torch.linalg.vector_norm(
            self.raw_direction.flatten(1), dim=1, keepdim=True
        ).reshape(-1, 1, 1).clamp_min(self.direction_eps)
        magnitude = self.max_anchor_norm * self.raw_magnitude.sigmoid()
        return magnitude * direction

    def diagnostics(self) -> dict[str, object]:
        anchor_norms = torch.linalg.vector_norm(
            self().detach().float().flatten(1), dim=1
        )
        residual_norms = torch.linalg.vector_norm(
            self.residuals().detach().float().flatten(1), dim=1
        )
        return {
            "anchor_norms": anchor_norms.cpu().tolist(),
            "max_anchor_norms": self.max_anchor_norm.detach().float().flatten().cpu().tolist(),
            "residual_norms": residual_norms.cpu().tolist(),
            "initial_was_zero": self.initial_was_zero.cpu().tolist(),
            "initial_was_clipped": self.initial_was_clipped.cpu().tolist(),
        }


def predicted_noise_cosine(
    edited_prediction: torch.Tensor,
    base_prediction: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    if edited_prediction.shape != base_prediction.shape:
        raise ValueError("Edited and base predictions must have matching shapes")
    if edited_prediction.ndim < 2:
        raise ValueError("Predictions must contain a batch dimension")
    return F.cosine_similarity(
        edited_prediction.float().flatten(1),
        base_prediction.float().flatten(1),
        dim=1,
        eps=eps,
    ).mean()


def _prediction_tensor(output) -> torch.Tensor:
    if isinstance(output, tuple):
        return output[0]
    if hasattr(output, "sample"):
        return output.sample
    if isinstance(output, torch.Tensor):
        return output
    raise TypeError(f"Unsupported U-Net output type: {type(output).__name__}")


def paired_predictions(
    base_unet: torch.nn.Module,
    parameter_overrides: Mapping[str, torch.Tensor],
    state: DiffusionState,
    *,
    hidden_states: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    kwargs = dict(state.unet_kwargs)
    kwargs.update(
        encoder_hidden_states=(
            state.target_hidden_states if hidden_states is None else hidden_states
        ),
        return_dict=False,
    )
    with torch.no_grad():
        base_prediction = _prediction_tensor(
            base_unet(state.model_input, state.timestep, **kwargs)
        )
    edited_prediction = _prediction_tensor(
        torch.func.functional_call(
            base_unet,
            dict(parameter_overrides),
            args=(state.model_input, state.timestep),
            kwargs=kwargs,
            strict=False,
        )
    )
    return base_prediction, edited_prediction


def prediction_metrics(
    base_prediction: torch.Tensor,
    edited_prediction: torch.Tensor,
    cosine_eps: float,
) -> dict[str, torch.Tensor]:
    cosine = predicted_noise_cosine(
        edited_prediction,
        base_prediction,
        eps=cosine_eps,
    )
    base_norms = torch.linalg.vector_norm(base_prediction.float().flatten(1), dim=1)
    edited_norms = torch.linalg.vector_norm(edited_prediction.float().flatten(1), dim=1)
    ratio = (edited_norms / base_norms.clamp_min(cosine_eps)).mean()
    mse = F.mse_loss(edited_prediction.float(), base_prediction.float())
    return {"cosine": cosine, "prediction_norm_ratio": ratio, "mse": mse}


def prediction_objective(
    target_cosine: torch.Tensor,
    null_cosine: torch.Tensor | None = None,
) -> torch.Tensor:
    """Minimize target similarity and, when supplied, maximize null similarity."""

    if null_cosine is None:
        return target_cosine
    return target_cosine - null_cosine


def state_prediction_metrics(
    base_unet: torch.nn.Module,
    parameter_overrides: Mapping[str, torch.Tensor],
    state: DiffusionState,
    cosine_eps: float,
    *,
    use_null_retain_loss: bool,
) -> dict[str, torch.Tensor]:
    base_prediction, edited_prediction = paired_predictions(
        base_unet, parameter_overrides, state
    )
    target_metrics = prediction_metrics(
        base_prediction, edited_prediction, cosine_eps
    )
    metrics = dict(target_metrics)
    null_cosine = None
    if use_null_retain_loss:
        if state.null_hidden_states is None:
            raise ValueError(
                "null_hidden_states is required when use_null_retain_loss is True"
            )
        base_null_prediction, edited_null_prediction = paired_predictions(
            base_unet,
            parameter_overrides,
            state,
            hidden_states=state.null_hidden_states,
        )
        null_metrics = prediction_metrics(
            base_null_prediction,
            edited_null_prediction,
            cosine_eps,
        )
        null_cosine = null_metrics["cosine"]
        metrics.update(
            null_cosine=null_cosine,
            null_prediction_norm_ratio=null_metrics["prediction_norm_ratio"],
            null_mse=null_metrics["mse"],
        )
    metrics["loss"] = prediction_objective(target_metrics["cosine"], null_cosine)
    return metrics


@torch.no_grad()
def evaluate_anchor(
    base_unet: torch.nn.Module,
    edit_state: DifferentiableLegacyEditState,
    anchor_embeddings: torch.Tensor,
    states: Sequence[DiffusionState],
    cosine_eps: float,
    *,
    use_null_retain_loss: bool = False,
) -> dict[str, float]:
    if not states:
        raise ValueError("Validation states must be non-empty")
    overrides, _ = edit_state.effective_parameters(anchor_embeddings)
    totals = {
        "cosine": 0.0,
        "prediction_norm_ratio": 0.0,
        "mse": 0.0,
        "loss": 0.0,
    }
    if use_null_retain_loss:
        totals.update(
            null_cosine=0.0,
            null_prediction_norm_ratio=0.0,
            null_mse=0.0,
        )
    for state in states:
        metrics = state_prediction_metrics(
            base_unet,
            overrides,
            state,
            cosine_eps,
            use_null_retain_loss=use_null_retain_loss,
        )
        for name in totals:
            totals[name] += float(metrics[name].item())
    return {name: value / len(states) for name, value in totals.items()}


def _snapshot(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().clone()
        for name, value in module.state_dict().items()
        if name in {"raw_direction", "raw_magnitude"}
    }


def optimize_anchor(
    base_unet: torch.nn.Module,
    edit_state: DifferentiableLegacyEditState,
    anchor_model: BoundedAnchor,
    training_state_factory: Callable[[], DiffusionState],
    validation_states: Sequence[DiffusionState],
    config: AnchorTrainingConfig,
    *,
    retain_validation_states: Sequence[DiffusionState] = (),
    metrics_callback: Callable[[dict[str, object]], None] | None = None,
) -> AnchorTrainingResult:
    """Optimize only the bounded anchor and restore the earliest best state."""

    config.validate()
    if any(parameter.requires_grad for parameter in base_unet.parameters()):
        raise ValueError("base_unet must be frozen before anchor optimization")
    if base_unet.training:
        raise ValueError("base_unet must be in eval mode before anchor optimization")
    optimizer = torch.optim.Adam(anchor_model.parameters(), lr=config.learning_rate)
    history: list[dict[str, object]] = []

    initial_validation = evaluate_anchor(
        base_unet,
        edit_state,
        anchor_model(),
        validation_states,
        config.cosine_eps,
        use_null_retain_loss=config.use_null_retain_loss,
    )
    best_step = 0
    best_validation_cosine = initial_validation["cosine"]
    best_validation_loss = initial_validation["loss"]
    best_validation_null_cosine = initial_validation.get("null_cosine")
    best_raw_state = _snapshot(anchor_model)
    initial_record: dict[str, object] = {
        "step": 0,
        "validation_cosine": initial_validation["cosine"],
        "validation_prediction_norm_ratio": initial_validation[
            "prediction_norm_ratio"
        ],
        "validation_mse": initial_validation["mse"],
        "validation_loss": initial_validation["loss"],
        **anchor_model.diagnostics(),
    }
    if config.use_null_retain_loss:
        initial_record.update(
            validation_null_cosine=initial_validation["null_cosine"],
            validation_null_prediction_norm_ratio=initial_validation[
                "null_prediction_norm_ratio"
            ],
            validation_null_mse=initial_validation["null_mse"],
        )
    if retain_validation_states:
        retain_metrics = evaluate_anchor(
            base_unet,
            edit_state,
            anchor_model(),
            retain_validation_states,
            config.cosine_eps,
        )
        initial_record.update(
            retain_validation_cosine=retain_metrics["cosine"],
            retain_validation_prediction_norm_ratio=retain_metrics[
                "prediction_norm_ratio"
            ],
            retain_validation_mse=retain_metrics["mse"],
        )
    history.append(initial_record)
    if metrics_callback is not None:
        metrics_callback(initial_record)

    for step in range(1, config.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        anchors = anchor_model()
        overrides, edit_diagnostics = edit_state.effective_parameters(anchors)
        diffusion_state = training_state_factory()
        metrics = state_prediction_metrics(
            base_unet,
            overrides,
            diffusion_state,
            config.cosine_eps,
            use_null_retain_loss=config.use_null_retain_loss,
        )
        loss = metrics["loss"]
        if not torch.isfinite(loss).item():
            raise FloatingPointError(f"Non-finite loss at step {step}")
        loss.backward()

        squared_gradient_norm = 0.0
        for name, parameter in anchor_model.named_parameters():
            if parameter.grad is None:
                raise RuntimeError(f"Anchor parameter {name} has no gradient")
            if not torch.isfinite(parameter.grad).all().item():
                raise FloatingPointError(f"Non-finite gradient for {name} at step {step}")
            squared_gradient_norm += float(parameter.grad.detach().float().square().sum().item())
        gradient_norm = math.sqrt(squared_gradient_norm)
        training_anchor_diagnostics = anchor_model.diagnostics()
        optimizer.step()
        for name, parameter in anchor_model.named_parameters():
            if not torch.isfinite(parameter).all().item():
                raise FloatingPointError(f"Non-finite anchor parameter {name} at step {step}")

        record: dict[str, object] = {
            "step": step,
            "training_cosine": float(metrics["cosine"].detach().item()),
            "training_prediction_norm_ratio": float(
                metrics["prediction_norm_ratio"].detach().item()
            ),
            "training_mse": float(metrics["mse"].detach().item()),
            "training_loss": float(loss.detach().item()),
            "gradient_norm": gradient_norm,
            "prompt": diffusion_state.prompt,
            "prefix_index": diffusion_state.prefix_index,
            "state_seed": diffusion_state.seed,
            **training_anchor_diagnostics,
            **edit_diagnostics,
        }
        if config.use_null_retain_loss:
            record.update(
                training_null_cosine=float(metrics["null_cosine"].detach().item()),
                training_null_prediction_norm_ratio=float(
                    metrics["null_prediction_norm_ratio"].detach().item()
                ),
                training_null_mse=float(metrics["null_mse"].detach().item()),
            )

        should_validate = step % config.validation_interval == 0 or step == config.steps
        if should_validate:
            validation = evaluate_anchor(
                base_unet,
                edit_state,
                anchor_model(),
                validation_states,
                config.cosine_eps,
                use_null_retain_loss=config.use_null_retain_loss,
            )
            record.update(
                validation_cosine=validation["cosine"],
                validation_prediction_norm_ratio=validation[
                    "prediction_norm_ratio"
                ],
                validation_mse=validation["mse"],
                validation_loss=validation["loss"],
            )
            if config.use_null_retain_loss:
                record.update(
                    validation_null_cosine=validation["null_cosine"],
                    validation_null_prediction_norm_ratio=validation[
                        "null_prediction_norm_ratio"
                    ],
                    validation_null_mse=validation["null_mse"],
                )
            if retain_validation_states:
                retain_metrics = evaluate_anchor(
                    base_unet,
                    edit_state,
                    anchor_model(),
                    retain_validation_states,
                    config.cosine_eps,
                )
                record.update(
                    retain_validation_cosine=retain_metrics["cosine"],
                    retain_validation_prediction_norm_ratio=retain_metrics[
                        "prediction_norm_ratio"
                    ],
                    retain_validation_mse=retain_metrics["mse"],
                )
            if validation["loss"] < best_validation_loss:
                best_validation_loss = validation["loss"]
                best_validation_cosine = validation["cosine"]
                best_validation_null_cosine = validation.get("null_cosine")
                best_step = step
                best_raw_state = _snapshot(anchor_model)

        history.append(record)
        if metrics_callback is not None:
            metrics_callback(record)

    anchor_model.load_state_dict(best_raw_state, strict=False)
    return AnchorTrainingResult(
        best_step=best_step,
        best_validation_cosine=best_validation_cosine,
        best_validation_loss=best_validation_loss,
        best_validation_null_cosine=best_validation_null_cosine,
        best_raw_state=best_raw_state,
        history=history,
    )


@torch.no_grad()
def sample_prefix_diffusion_state(
    pipe,
    target_hidden_states: torch.Tensor,
    null_hidden_states: torch.Tensor,
    *,
    prompt: str,
    seed: int,
    prefix_index: int,
    num_inference_steps: int,
    guidance_scale: float,
    resolution: int,
    generator: torch.Generator,
) -> DiffusionState:
    """Sample one frozen SD trajectory prefix and return its conditional input."""

    if num_inference_steps <= 0:
        raise ValueError("num_inference_steps must be positive")
    if not 0 <= prefix_index < num_inference_steps:
        raise ValueError("prefix_index is outside the scheduler range")
    if resolution <= 0:
        raise ValueError("resolution must be positive")

    scheduler = pipe.scheduler.__class__.from_config(pipe.scheduler.config)
    scheduler.set_timesteps(num_inference_steps, device=pipe.unet.device)
    timesteps = scheduler.timesteps
    batch_size = target_hidden_states.shape[0]
    if null_hidden_states.shape[0] == 1 and batch_size > 1:
        null_hidden_states = null_hidden_states.expand(batch_size, -1, -1)
    if null_hidden_states.shape[0] != batch_size:
        raise ValueError("Null and target hidden-state batch sizes must match")

    latents = pipe.prepare_latents(
        batch_size,
        pipe.unet.config.in_channels,
        resolution,
        resolution,
        target_hidden_states.dtype,
        pipe.unet.device,
        generator,
        None,
    )
    extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator, 0.0)
    guidance_hidden_states = torch.cat([null_hidden_states, target_hidden_states])
    for timestep in timesteps[:prefix_index]:
        latent_model_input = torch.cat([latents, latents])
        latent_model_input = scheduler.scale_model_input(latent_model_input, timestep)
        noise_prediction = _prediction_tensor(
            pipe.unet(
                latent_model_input,
                timestep,
                encoder_hidden_states=guidance_hidden_states,
                return_dict=False,
            )
        )
        noise_unconditional, noise_conditional = noise_prediction.chunk(2)
        guided_prediction = noise_unconditional + guidance_scale * (
            noise_conditional - noise_unconditional
        )
        latents = scheduler.step(
            guided_prediction,
            timestep,
            latents,
            **extra_step_kwargs,
            return_dict=False,
        )[0]

    evaluation_timestep = timesteps[prefix_index]
    model_input = scheduler.scale_model_input(latents, evaluation_timestep)
    return DiffusionState(
        model_input=model_input.detach(),
        timestep=evaluation_timestep.detach().clone(),
        target_hidden_states=target_hidden_states.detach(),
        null_hidden_states=null_hidden_states.detach(),
        prompt=prompt,
        prefix_index=prefix_index,
        seed=seed,
    )
