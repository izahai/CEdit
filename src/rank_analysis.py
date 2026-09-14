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
    targets = _as_matrix(target_embeddings, "target_embeddings")
    legacy = _as_matrix(legacy_residuals, "legacy_residuals")
    basis = _as_matrix(basis, "basis")
    if targets.shape != legacy.shape or basis.shape[1] != targets.shape[1]:
        raise ValueError("Incompatible target, residual, or basis shapes")
    if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
        raise ValueError("rank must be a positive integer")
    realized_basis_rank = min(rank, basis.shape[0])
    selected = basis[:realized_basis_rank]
    projected_targets = (targets @ selected.T) @ selected
    projected_legacy = (legacy @ selected.T) @ selected
    target_norms = torch.linalg.vector_norm(projected_targets, dim=1)
    legacy_norms = torch.linalg.vector_norm(projected_legacy, dim=1)
    directions = -projected_targets
    target_fallback = target_norms <= eps
    use_legacy = target_fallback & (legacy_norms > eps)
    use_basis = target_fallback & ~use_legacy
    directions[use_legacy] = projected_legacy[use_legacy]
    directions[use_basis] = selected[0]
    residuals = norm_match_rows(directions, legacy, eps=eps, fallback_basis=selected)
    return residuals, {
        "requested_rank": int(rank),
        "basis_rank": int(realized_basis_rank),
        "target_projection_fallback_count": int(target_fallback.sum()),
        "legacy_fallback_count": int(use_legacy.sum()),
        "basis_fallback_count": int(use_basis.sum()),
    }


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
    return {
        "target_effect": float(target_effect.mean()),
        "target_rotation_deg": float(
            torch.rad2deg(torch.acos(rotation_cosine)).mean()
        ),
        "retain_leakage": float(retain_leakage.mean()),
        "directional_realization_cosine": float(directional_cosine.mean()),
        "directional_relative_error": float(directional_error.mean()),
        "delta_frobenius_norm": float(torch.linalg.vector_norm(delta)),
    }


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
    )
    return {
        f"{name}_mean": sum(float(row[name]) for row in rows) / len(rows)
        for name in metric_names
    }
