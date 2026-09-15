"""Numerical primitives for residual-rank and SPEED update analysis."""

import torch

from src.residual_subspace import build_global_pairwise_residual_matrix


def _as_matrix(values, name):
    if values.ndim < 2:
        raise ValueError(f"{name} must include a batch dimension")
    matrix = values.reshape(values.shape[0], -1).float()
    if not torch.isfinite(matrix).all():
        raise ValueError(f"{name} contains non-finite values")
    return matrix


def normalize_rows(matrix, eps=1e-8):
    """Normalize each row while leaving zero rows equal to zero."""
    matrix = _as_matrix(matrix, "matrix")
    norms = torch.linalg.vector_norm(matrix, dim=1, keepdim=True)
    return matrix / norms.clamp_min(eps)


def spectral_metrics(matrix, rtol=1e-5, atol=0.0, singular_values=None):
    """Return hard and energy-sensitive rank metrics for a matrix."""
    matrix = _as_matrix(matrix, "matrix")
    if rtol < 0 or atol < 0:
        raise ValueError("Rank tolerances must be non-negative")
    values = (
        torch.linalg.svdvals(matrix)
        if singular_values is None
        else singular_values.float()
    )
    if values.ndim != 1:
        raise ValueError("singular_values must be one-dimensional")
    sigma_max = values[0] if values.numel() else matrix.new_tensor(0.0)
    tolerance = max(float(atol), float(rtol) * float(sigma_max))
    energy = values.square()
    total_energy = energy.sum()
    if float(total_energy) == 0.0:
        stable_rank = 0.0
        effective_rank = 0.0
    else:
        stable_rank = float(total_energy / sigma_max.square())
        probabilities = energy / total_energy
        positive = probabilities > 0
        entropy = -(probabilities[positive] * probabilities[positive].log()).sum()
        effective_rank = float(entropy.exp())
    return {
        "numerical_rank": int((values > tolerance).sum()),
        "stable_rank": stable_rank,
        "effective_rank": effective_rank,
        "frobenius_norm": float(torch.sqrt(total_energy)),
        "spectral_norm": float(sigma_max),
        "rank_tolerance": tolerance,
    }, values


def norm_match_rows(candidate, reference, eps=1e-8, fallback_basis=None):
    """Match candidate row norms to reference row norms without changing rank."""
    candidate = _as_matrix(candidate, "candidate")
    reference = _as_matrix(reference, "reference")
    if candidate.shape != reference.shape:
        raise ValueError("candidate and reference must have the same shape")
    candidate_norms = torch.linalg.vector_norm(candidate, dim=1, keepdim=True)
    reference_norms = torch.linalg.vector_norm(reference, dim=1, keepdim=True)
    near_zero = candidate_norms.squeeze(1) <= eps
    directions = candidate / candidate_norms.clamp_min(eps)
    if near_zero.any():
        if fallback_basis is None:
            fallback = normalize_rows(reference[near_zero], eps=eps)
        else:
            fallback_matrix = _as_matrix(fallback_basis, "fallback_basis")
            fallback = fallback_matrix[[0]].expand(int(near_zero.sum()), -1)
            fallback = normalize_rows(fallback, eps=eps)
        directions[near_zero] = fallback
    return directions * reference_norms


def truncated_svd_residuals(legacy_residuals, rank, eps=1e-8):
    """Construct a per-row norm-matched rank-q truncation of legacy residuals."""
    legacy = _as_matrix(legacy_residuals, "legacy_residuals")
    if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
        raise ValueError("rank must be a positive integer")
    if rank > min(legacy.shape):
        raise ValueError(f"rank {rank} exceeds maximum rank {min(legacy.shape)}")
    u, singular_values, vh = torch.linalg.svd(legacy, full_matrices=False)
    truncated = (u[:, :rank] * singular_values[:rank]) @ vh[:rank]
    return norm_match_rows(truncated, legacy, eps=eps, fallback_basis=vh[:rank])


def random_rank_residuals(legacy_residuals, rank, seed, eps=1e-8):
    """Project residuals onto a seeded random rank-q subspace and norm-match."""
    legacy = _as_matrix(legacy_residuals, "legacy_residuals")
    if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
        raise ValueError("rank must be a positive integer")
    if rank > min(legacy.shape):
        raise ValueError(f"rank {rank} exceeds maximum rank {min(legacy.shape)}")
    generator = torch.Generator(device=legacy.device)
    generator.manual_seed(int(seed))
    random_matrix = torch.randn(
        legacy.shape[1], rank, generator=generator, device=legacy.device
    )
    basis = torch.linalg.qr(random_matrix, mode="reduced").Q.T
    projected = (legacy @ basis.T) @ basis
    return norm_match_rows(projected, legacy, eps=eps, fallback_basis=basis)


def projected_target_residuals(
    target_embeddings,
    legacy_residuals,
    basis,
    rank,
    sign=-1.0,
    complement=False,
    eps=1e-8,
):
    """Build norm-matched residuals from signed target projections."""
    targets = _as_matrix(target_embeddings, "target_embeddings")
    legacy = _as_matrix(legacy_residuals, "legacy_residuals")
    basis = _as_matrix(basis, "basis")
    if targets.shape != legacy.shape or basis.shape[1] != targets.shape[1]:
        raise ValueError("Incompatible target, residual, or basis shapes")
    if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
        raise ValueError("rank must be a positive integer")
    if sign not in (-1.0, 1.0):
        raise ValueError("sign must be either -1.0 or 1.0")
    realized_basis_rank = min(rank, basis.shape[0])
    selected = basis[:realized_basis_rank]
    projected_targets = (targets @ selected.T) @ selected
    projected_legacy = (legacy @ selected.T) @ selected
    primary = targets - projected_targets if complement else projected_targets
    fallback_legacy = legacy - projected_legacy if complement else projected_legacy
    primary_norms = torch.linalg.vector_norm(primary, dim=1)
    legacy_norms = torch.linalg.vector_norm(fallback_legacy, dim=1)
    directions = float(sign) * primary
    primary_fallback = primary_norms <= eps
    use_legacy = primary_fallback & (legacy_norms > eps)
    use_basis = primary_fallback & ~use_legacy
    directions[use_legacy] = float(sign) * fallback_legacy[use_legacy]
    fallback = selected
    if complement:
        projector_diagonal = 1.0 - selected.square().sum(dim=0)
        coordinate = int(projector_diagonal.argmax())
        complement_vector = targets.new_zeros(targets.shape[1])
        complement_vector[coordinate] = 1.0
        complement_vector = complement_vector - (
            (complement_vector @ selected.T) @ selected
        )
        fallback = complement_vector.unsqueeze(0)
    directions[use_basis] = float(sign) * fallback[0]
    residuals = norm_match_rows(
        directions, legacy, eps=eps, fallback_basis=fallback
    )
    return residuals, {
        "requested_rank": int(rank),
        "basis_rank": int(realized_basis_rank),
        "projection_sign": float(sign),
        "uses_complement": bool(complement),
        "target_projection_fallback_count": int(primary_fallback.sum()),
        "legacy_fallback_count": int(use_legacy.sum()),
        "basis_fallback_count": int(use_basis.sum()),
    }


def random_negative_target_residuals(
    target_embeddings, legacy_residuals, rank, seed, eps=1e-8
):
    """Apply -P_random t with a seeded rank-q orthonormal basis."""
    targets = _as_matrix(target_embeddings, "target_embeddings")
    generator = torch.Generator(device=targets.device)
    generator.manual_seed(int(seed))
    random_matrix = torch.randn(
        targets.shape[1], rank, generator=generator, device=targets.device
    )
    basis = torch.linalg.qr(random_matrix, mode="reduced").Q.T
    residuals, diagnostics = projected_target_residuals(
        targets, legacy_residuals, basis, rank, sign=-1.0, eps=eps
    )
    diagnostics["basis_seed"] = int(seed)
    return residuals, diagnostics


def build_tgprs_basis(target_embeddings, extra_anchor_embeddings, max_rank, eps=1e-8):
    """Build one normalized pairwise basis that can be sliced for many ranks."""
    targets = _as_matrix(target_embeddings, "target_embeddings")
    anchors = _as_matrix(extra_anchor_embeddings, "extra_anchor_embeddings")
    pairwise = build_global_pairwise_residual_matrix(targets, anchors)
    normalized = normalize_rows(pairwise, eps=eps)
    _, singular_values, vh = torch.linalg.svd(normalized, full_matrices=False)
    tolerance = max(normalized.shape) * torch.finfo(normalized.dtype).eps * singular_values[0]
    effective_rank = int((singular_values > tolerance).sum())
    if effective_rank == 0:
        raise ValueError("Pairwise residual matrix has zero numerical rank")
    basis_rank = min(int(max_rank), effective_rank)
    return vh[:basis_rank], singular_values, {
        "pairwise_residual_count": int(pairwise.shape[0]),
        "pairwise_residual_shape": list(pairwise.shape),
        "pairwise_effective_rank": effective_rank,
        "basis_rank": basis_rank,
    }


def tgprs_residuals_from_basis(
    target_embeddings,
    legacy_residuals,
    basis,
    rank,
    eps=1e-8,
):
    """Apply the TGPRS direction -P_S t with legacy per-target magnitudes."""
    return projected_target_residuals(
        target_embeddings,
        legacy_residuals,
        basis,
        rank,
        sign=-1.0,
        complement=False,
        eps=eps,
    )


def edit_statistic(residuals, targets):
    residuals = _as_matrix(residuals, "residuals")
    targets = _as_matrix(targets, "targets")
    if residuals.shape != targets.shape:
        raise ValueError("residuals and targets must have the same shape")
    return residuals.T @ targets / targets.shape[0]


def second_moment(embeddings):
    embeddings = _as_matrix(embeddings, "embeddings")
    return embeddings.T @ embeddings / embeddings.shape[0]


def retain_low_projection(retain_embeddings, threshold):
    """Match SPEED's absolute-singular-value retain projector."""
    covariance = second_moment(retain_embeddings)
    u, singular_values, _ = torch.linalg.svd(covariance)
    mask = singular_values < float(threshold)
    projection = u[:, mask] @ u[:, mask].T
    return projection, singular_values, int(mask.sum())


def retain_eigensystem(retain_embeddings):
    """Return SPEED's retain covariance eigenvectors and singular values."""
    covariance = second_moment(retain_embeddings)
    u, singular_values, _ = torch.linalg.svd(covariance)
    return u, singular_values


def retain_projection_from_eigensystem(u, singular_values, threshold):
    """Construct a retain-low projector from a cached eigensystem."""
    mask = singular_values < float(threshold)
    projection = u[:, mask] @ u[:, mask].T
    return projection, int(mask.sum())


def speed_retain_construction(
    retain_embeddings,
    layer_weight,
    statistic,
    target_second_moment,
    aug_num=10,
    filter_enabled=True,
    seed=0,
    eps=1e-8,
):
    """Reproduce SPEED's layer-specific hard-retain construction."""
    retain = _as_matrix(retain_embeddings, "retain_embeddings")
    weight = layer_weight.float()
    statistic = statistic.float()
    target_second_moment = target_second_moment.float()
    identity = torch.eye(target_second_moment.shape[0], device=retain.device)
    erase_weight = weight @ statistic @ torch.linalg.inv(
        identity + target_second_moment
    )
    responses = torch.linalg.vector_norm(retain @ erase_weight.T, dim=1)
    if filter_enabled:
        selected = retain[responses > responses.mean()]
    else:
        selected = retain
    if selected.shape[0] == 0:
        raise ValueError("SPEED retain filtering removed every retain embedding")

    augmented = selected.new_empty((0, selected.shape[1]))
    if int(aug_num) > 0:
        _, _, vh = torch.linalg.svd(weight, full_matrices=False)
        weakest = vh[-1]
        generator = torch.Generator(device=selected.device)
        generator.manual_seed(int(seed))
        candidates = []
        for _ in range(int(aug_num)):
            noise = torch.randn(
                selected.shape,
                generator=generator,
                device=selected.device,
                dtype=selected.dtype,
            )
            perturbation = (noise @ weakest).unsqueeze(1) * weakest.unsqueeze(0)
            candidates.append(selected + perturbation)
        candidates = torch.cat(candidates, dim=0)
        candidate_responses = torch.linalg.vector_norm(
            candidates @ erase_weight.T, dim=1
        )
        augmented = candidates[
            candidate_responses > candidate_responses.mean().clamp_min(eps)
        ]
    constructed = torch.cat([selected, augmented], dim=0)
    return constructed, {
        "retain_original_count": int(retain.shape[0]),
        "retain_filtered_count": int(selected.shape[0]),
        "retain_augmented_count": int(augmented.shape[0]),
        "retain_constructed_count": int(constructed.shape[0]),
        "retain_aug_num": int(aug_num),
        "retain_filter_enabled": bool(filter_enabled),
        "retain_seed": int(seed),
    }


def speed_right_factor(
    target_second_moment,
    retain_projection,
    k2,
    retain_scale,
    lamb=0.0,
):
    """Return the residual-independent right factor in SPEED's update."""
    target_second_moment = target_second_moment.float()
    retain_projection = retain_projection.float()
    k2 = k2.float()
    dimension = target_second_moment.shape[0]
    identity = torch.eye(dimension, device=target_second_moment.device)
    middle_identity = torch.eye(k2.shape[1], device=target_second_moment.device)
    m = torch.linalg.inv(
        target_second_moment @ retain_projection + float(retain_scale) * identity
    )
    correction_inverse = torch.linalg.inv(
        k2.T @ retain_projection @ m @ k2 + float(lamb) * middle_identity
    )
    correction = (
        identity
        - m @ k2 @ correction_inverse @ k2.T @ retain_projection
    )
    return retain_projection @ correction @ m


def speed_delta_weight(layer_weight, statistic, right_factor):
    return layer_weight.float() @ statistic.float() @ right_factor.float()


def speed_delta_weight_low_rank(layer_weight, residuals, targets, right_factor):
    """Evaluate W R^T T A / N without materializing the d-by-d statistic."""
    weight = layer_weight.float()
    residuals = _as_matrix(residuals, "residuals")
    targets = _as_matrix(targets, "targets")
    if residuals.shape != targets.shape:
        raise ValueError("residuals and targets must have the same shape")
    transformed_targets = (
        targets @ right_factor.float() / targets.shape[0]
    )
    return speed_delta_weight_from_target_factor(
        weight, residuals, transformed_targets
    )


def speed_delta_weight_from_target_factor(
    layer_weight, residuals, transformed_targets
):
    """Evaluate (W R^T)(T A / N) with a cached target-side factor."""
    weight = layer_weight.float()
    residuals = _as_matrix(residuals, "residuals")
    transformed = _as_matrix(transformed_targets, "transformed_targets")
    if transformed.shape[0] != residuals.shape[0]:
        raise ValueError("residuals and transformed_targets must share a batch")
    return (weight @ residuals.T) @ transformed


def low_rank_delta_spectrum(layer_weight, residuals, targets, right_factor):
    """Compute singular values of W R^T T A / N through a small core."""
    residuals = _as_matrix(residuals, "residuals")
    targets = _as_matrix(targets, "targets")
    if residuals.shape != targets.shape:
        raise ValueError("residuals and targets must have the same shape")
    transformed_targets = (
        targets @ right_factor.float() / targets.shape[0]
    )
    return low_rank_delta_spectrum_from_target_factor(
        layer_weight, residuals, transformed_targets
    )


def low_rank_delta_spectrum_from_target_factor(
    layer_weight, residuals, transformed_targets
):
    """Compute update singular values with a cached T A / N factor."""
    weight = layer_weight.float()
    residuals = _as_matrix(residuals, "residuals")
    transformed = _as_matrix(transformed_targets, "transformed_targets")
    if transformed.shape[0] != residuals.shape[0]:
        raise ValueError("residuals and transformed_targets must share a batch")
    left = weight @ residuals.T
    right = transformed.T
    _, left_r = torch.linalg.qr(left, mode="reduced")
    _, right_r = torch.linalg.qr(right, mode="reduced")
    return torch.linalg.svdvals(left_r @ right_r.T)


def _row_cosine(left, right, eps):
    numerator = (left * right).sum(dim=1)
    denominator = (
        torch.linalg.vector_norm(left, dim=1)
        * torch.linalg.vector_norm(right, dim=1)
    )
    return numerator / denominator.clamp_min(eps)


def layer_edit_metrics(
    layer_weight,
    delta_weight,
    targets,
    residuals,
    retain_embeddings,
    anchor_embeddings=None,
    canonical_residuals=None,
    eps=1e-8,
):
    """Measure target intervention, retain leakage, and residual realization."""
    weight = layer_weight.float()
    delta = delta_weight.float()
    targets = _as_matrix(targets, "targets")
    residuals = _as_matrix(residuals, "residuals")
    retain = _as_matrix(retain_embeddings, "retain_embeddings")
    base_target = targets @ weight.T
    target_change = targets @ delta.T
    edited_target = base_target + target_change
    desired_change = residuals @ weight.T
    base_retain = retain @ weight.T
    retain_change = retain @ delta.T

    target_effect = (
        torch.linalg.vector_norm(target_change, dim=1)
        / torch.linalg.vector_norm(base_target, dim=1).clamp_min(eps)
    )
    retain_leakage = (
        torch.linalg.vector_norm(retain_change, dim=1)
        / torch.linalg.vector_norm(base_retain, dim=1).clamp_min(eps)
    )
    rotation_cosine = _row_cosine(base_target, edited_target, eps).clamp(-1.0, 1.0)
    directional_cosine = _row_cosine(target_change, desired_change, eps)
    directional_error = (
        torch.linalg.vector_norm(target_change - desired_change, dim=1)
        / torch.linalg.vector_norm(desired_change, dim=1).clamp_min(eps)
    )
    metrics = {
        "target_effect": float(target_effect.mean()),
        "target_rotation_deg": float(
            torch.rad2deg(torch.acos(rotation_cosine)).mean()
        ),
        "retain_leakage": float(retain_leakage.mean()),
        "directional_realization_cosine": float(directional_cosine.mean()),
        "directional_relative_error": float(directional_error.mean()),
        "delta_frobenius_norm": float(torch.linalg.vector_norm(delta)),
    }
    metrics["edited_output_norm_ratio"] = float(
        (
            torch.linalg.vector_norm(edited_target, dim=1)
            / torch.linalg.vector_norm(base_target, dim=1).clamp_min(eps)
        ).mean()
    )
    if canonical_residuals is not None:
        canonical = _as_matrix(canonical_residuals, "canonical_residuals")
        canonical_change = canonical @ weight.T
        metrics["canonical_erasure_alignment"] = float(
            _row_cosine(target_change, canonical_change, eps).mean()
        )
    if anchor_embeddings is not None:
        anchors = _as_matrix(anchor_embeddings, "anchor_embeddings")
        base_anchor = anchors @ weight.T
        metrics["anchor_distance_ratio"] = float(
            (
                torch.linalg.vector_norm(edited_target - base_anchor, dim=1)
                / torch.linalg.vector_norm(base_target - base_anchor, dim=1).clamp_min(eps)
            ).mean()
        )
    return metrics


def aggregate_layer_metrics(rows):
    """Average scalar layer metrics without coupling to pandas."""
    if not rows:
        raise ValueError("At least one layer row is required")
    metric_names = (
        "target_effect",
        "target_rotation_deg",
        "retain_leakage",
        "directional_realization_cosine",
        "directional_relative_error",
        "delta_frobenius_norm",
        "edited_output_norm_ratio",
        "canonical_erasure_alignment",
        "anchor_distance_ratio",
    )
    return {
        f"{name}_mean": sum(float(row[name]) for row in rows) / len(rows)
        for name in metric_names
        if all(name in row for row in rows)
    }
