import torch


def _project_onto_basis(vectors, basis):
    return (vectors @ basis.T) @ basis


def _cosine_with_shift(targets, residuals, eps):
    shifted_targets = targets + residuals
    numerator = (targets * shifted_targets).sum(dim=1)
    denominator = (
        torch.linalg.vector_norm(targets, dim=1)
        * torch.linalg.vector_norm(shifted_targets, dim=1)
    )
    return numerator / denominator.clamp_min(eps)


def build_smallest_cosine_subspace_residuals(
    target_embeddings,
    anchor_embeddings,
    top_k,
    eps=1e-8,
):
    """Build norm-matched residuals opposite each target's subspace projection."""
    if target_embeddings.ndim < 2:
        raise ValueError("Target embeddings must include a batch dimension")
    if target_embeddings.shape != anchor_embeddings.shape:
        raise ValueError(
            "Target and anchor embedding shapes must match, got "
            f"{tuple(target_embeddings.shape)} and {tuple(anchor_embeddings.shape)}"
        )
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError(f"Residual top-k must be a positive integer, got {top_k}")

    pair_count = target_embeddings.shape[0]
    if top_k > pair_count:
        raise ValueError(
            f"Residual top-k {top_k} exceeds the number of target-anchor pairs "
            f"({pair_count})"
        )
    if not isinstance(eps, (int, float)) or eps <= 0:
        raise ValueError(f"Epsilon must be positive, got {eps}")

    output_dtype = target_embeddings.dtype
    legacy_residuals = anchor_embeddings - target_embeddings
    flattened_targets = target_embeddings.reshape(pair_count, -1).float()
    flattened_legacy = legacy_residuals.reshape(pair_count, -1).float()
    legacy_norms = torch.linalg.vector_norm(flattened_legacy, dim=1)

    normalized_residuals = flattened_legacy / legacy_norms.unsqueeze(1).clamp_min(eps)
    cosine_similarity = normalized_residuals @ normalized_residuals.T
    if pair_count > 1:
        mean_cosine_scores = (
            cosine_similarity.sum(dim=1) - cosine_similarity.diagonal()
        ) / (pair_count - 1)
    else:
        mean_cosine_scores = cosine_similarity.diagonal()

    selected_indices = torch.argsort(mean_cosine_scores, stable=True)[:top_k]
    selected_residuals = flattened_legacy[selected_indices]
    _, singular_values, vh = torch.linalg.svd(selected_residuals, full_matrices=False)
    largest_singular_value = singular_values.max()
    rank_tolerance = (
        max(selected_residuals.shape)
        * torch.finfo(singular_values.dtype).eps
        * largest_singular_value
    )
    effective_rank = int((singular_values > rank_tolerance).sum().item())
    if effective_rank == 0:
        raise ValueError(
            "The selected residuals span a zero-rank subspace; cannot construct "
            "a unit residual direction"
        )
    basis = vh[:effective_rank]

    projected_targets = _project_onto_basis(flattened_targets, basis)
    projected_target_norms = torch.linalg.vector_norm(projected_targets, dim=1)
    projected_legacy = _project_onto_basis(flattened_legacy, basis)
    projected_legacy_norms = torch.linalg.vector_norm(projected_legacy, dim=1)

    unit_directions = torch.empty_like(flattened_targets)
    target_projection_mask = projected_target_norms > eps
    legacy_fallback_mask = ~target_projection_mask & (projected_legacy_norms > eps)
    basis_fallback_mask = ~target_projection_mask & ~legacy_fallback_mask

    unit_directions[target_projection_mask] = (
        -projected_targets[target_projection_mask]
        / projected_target_norms[target_projection_mask].unsqueeze(1)
    )
    unit_directions[legacy_fallback_mask] = (
        projected_legacy[legacy_fallback_mask]
        / projected_legacy_norms[legacy_fallback_mask].unsqueeze(1)
    )
    unit_directions[basis_fallback_mask] = basis[0]

    flattened_residuals = legacy_norms.unsqueeze(1) * unit_directions
    projected_new_residuals = _project_onto_basis(flattened_residuals, basis)
    subspace_errors = torch.linalg.vector_norm(
        flattened_residuals - projected_new_residuals,
        dim=1,
    )
    norm_errors = (
        torch.linalg.vector_norm(flattened_residuals, dim=1) - legacy_norms
    ).abs()
    legacy_cosines = _cosine_with_shift(
        flattened_targets,
        flattened_legacy,
        eps,
    )
    new_cosines = _cosine_with_shift(
        flattened_targets,
        flattened_residuals,
        eps,
    )

    residuals = flattened_residuals.reshape_as(legacy_residuals).to(output_dtype)
    diagnostics = {
        "subspace_requested_top_k": int(top_k),
        "subspace_selected_indices": selected_indices.tolist(),
        "subspace_selected_scores": mean_cosine_scores[selected_indices].tolist(),
        "subspace_effective_rank": effective_rank,
        "subspace_singular_values": singular_values[:5].tolist(),
        "subspace_legacy_cosine_mean": legacy_cosines.mean().item(),
        "subspace_new_cosine_mean": new_cosines.mean().item(),
        "subspace_max_norm_error": norm_errors.max().item(),
        "subspace_max_projection_error": subspace_errors.max().item(),
        "subspace_target_projection_fallback_count": int(
            (~target_projection_mask).sum().item()
        ),
        "subspace_legacy_fallback_count": int(legacy_fallback_mask.sum().item()),
        "subspace_basis_fallback_count": int(basis_fallback_mask.sum().item()),
    }
    return residuals, diagnostics
