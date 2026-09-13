"""Validated loading and saving for partial U-Net edit checkpoints."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

import torch

from src.differentiable_legacy_edit import SPEED_CHECKPOINT_FORMAT


def _validate_tensors(
    tensors: Mapping[str, torch.Tensor],
    reference_state: Mapping[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    if not isinstance(tensors, Mapping) or not tensors:
        raise ValueError("Edit checkpoint must contain a non-empty tensor mapping")
    validated: dict[str, torch.Tensor] = {}
    for name, tensor in tensors.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Edit checkpoint keys must be non-empty strings")
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Edit checkpoint value for {name} is not a tensor")
        if not torch.is_floating_point(tensor):
            raise TypeError(f"Edit checkpoint tensor {name} must be floating point")
        if not torch.isfinite(tensor).all().item():
            raise ValueError(f"Edit checkpoint tensor {name} contains non-finite values")
        if reference_state is not None:
            if name not in reference_state:
                raise KeyError(f"Edit checkpoint contains unexpected parameter: {name}")
            expected = reference_state[name]
            if tensor.shape != expected.shape:
                raise ValueError(
                    f"Shape mismatch for {name}: checkpoint {tuple(tensor.shape)}, "
                    f"model {tuple(expected.shape)}"
                )
        validated[name] = tensor
    return validated


def save_speed_checkpoint(
    tensors: Mapping[str, torch.Tensor],
    filename: str | os.PathLike[str],
    metadata: Mapping[str, object] | None = None,
) -> None:
    path = Path(filename)
    if path.suffix != ".safetensors":
        raise ValueError("Differentiable SPEED checkpoints must use .safetensors")
    try:
        from safetensors.torch import save_file
    except ImportError as error:
        raise RuntimeError("safetensors is required to save this checkpoint") from error

    values = _validate_tensors(tensors)
    serialized = {
        name: tensor.detach().to(device="cpu").contiguous()
        for name, tensor in values.items()
    }
    final_metadata = {}
    if metadata:
        final_metadata.update(
            {name: str(value) for name, value in metadata.items() if value is not None}
        )
    final_metadata["format"] = SPEED_CHECKPOINT_FORMAT
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(serialized, str(path), metadata=final_metadata)


def load_edit_checkpoint(
    filename: str | os.PathLike[str],
    *,
    device: str | torch.device = "cpu",
    reference_state: Mapping[str, torch.Tensor] | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    path = Path(filename)
    if not path.is_file():
        raise FileNotFoundError(f"Edit checkpoint not found: {path}")

    metadata: dict[str, str] = {}
    if path.suffix == ".safetensors":
        try:
            from safetensors import safe_open
            from safetensors.torch import load_file
        except ImportError as error:
            raise RuntimeError("safetensors is required to load this checkpoint") from error
        with safe_open(str(path), framework="pt", device=str(device)) as handle:
            metadata = handle.metadata() or {}
        if metadata.get("format") != SPEED_CHECKPOINT_FORMAT:
            raise ValueError(
                "Unsupported safetensors edit checkpoint format: "
                f"{metadata.get('format')!r}"
            )
        tensors = load_file(str(path), device=str(device))
    elif path.suffix in {".pt", ".pth"}:
        try:
            tensors = torch.load(path, map_location=device, weights_only=True)
        except TypeError:
            tensors = torch.load(path, map_location=device)
    else:
        raise ValueError(
            f"Unsupported edit checkpoint extension {path.suffix!r}; expected .pt, .pth, or .safetensors"
        )

    return _validate_tensors(tensors, reference_state), metadata


def apply_edit_checkpoint(
    module: torch.nn.Module,
    filename: str | os.PathLike[str],
    *,
    device: str | torch.device = "cpu",
):
    tensors, metadata = load_edit_checkpoint(
        filename,
        device=device,
        reference_state=module.state_dict(),
    )
    result = module.load_state_dict(tensors, strict=False)
    if result.unexpected_keys:
        raise ValueError(
            "Edit checkpoint contains unexpected keys: "
            + ", ".join(result.unexpected_keys)
        )
    return metadata, result
