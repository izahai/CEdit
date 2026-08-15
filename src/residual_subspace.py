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


def _cosine_between(left, right, eps):
    numerator = (left * right).sum(dim=1)
    denominator = (
        torch.linalg.vector_norm(left, dim=1)
        * torch.linalg.vector_norm(right, dim=1)
    )
    return numerator / denominator.clamp_min(eps)


def build_global_pairwise_residual_matrix(
    concept_embeddings,
    extra_anchor_embeddings,
):
    """Build ordered concept residuals followed by each concept's anchors."""
    if concept_embeddings.ndim < 2:
        raise ValueError("Concept embeddings must include a batch dimension")
    if extra_anchor_embeddings.ndim != concept_embeddings.ndim:
        raise ValueError(
            "Concept and extra-anchor embeddings must have the same number of "
            "dimensions"
        )
    if concept_embeddings.shape[0] == 0:
        raise ValueError("At least one concept embedding is required")
    if extra_anchor_embeddings.shape[0] != 2:
        raise ValueError(
            "Exactly two extra-anchor embeddings are required: empty prompt and "
            "person"
        )
    if concept_embeddings.shape[1:] != extra_anchor_embeddings.shape[1:]:
        raise ValueError(
            "Concept and extra-anchor embedding shapes must match, got "
            f"{tuple(concept_embeddings.shape[1:])} and "
            f"{tuple(extra_anchor_embeddings.shape[1:])}"
        )

    flattened_concepts = concept_embeddings.reshape(
        concept_embeddings.shape[0], -1
    ).float()
    flattened_anchors = extra_anchor_embeddings.reshape(2, -1).float()
    residual_blocks = []
    for concept_index, source_embedding in enumerate(flattened_concepts):
        other_concepts = torch.cat(
            [
                flattened_concepts[:concept_index],
                flattened_concepts[concept_index + 1:],
            ],
            dim=0,
        )
        destinations = torch.cat([other_concepts, flattened_anchors], dim=0)
        residual_blocks.append(destinations - source_embedding)
    return torch.cat(residual_blocks, dim=0)


def _build_normalized_residual_basis(residual_matrix, rank, eps):
    if residual_matrix.ndim != 2:
        raise ValueError("Residual matrix must be two-dimensional")
    if not isinstance(eps, (int, float)) or eps <= 0:
        raise ValueError(f"Epsilon must be positive, got {eps}")
    if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
        raise ValueError(f"Residual rank must be a positive integer, got {rank}")
    if rank > min(residual_matrix.shape):
        raise ValueError(
            f"Residual rank {rank} exceeds the maximum possible rank "
            f"{min(residual_matrix.shape)} for residual matrix shape "
            f"{tuple(residual_matrix.shape)}"
        )

    residual_norms = torch.linalg.vector_norm(residual_matrix, dim=1)
    normalized_residuals = residual_matrix / residual_norms.unsqueeze(1).clamp_min(
        eps
    )
    _, singular_values, vh = torch.linalg.svd(
        normalized_residuals,
        full_matrices=False,
    )
    largest_singular_value = singular_values.max()
    rank_tolerance = (
        max(normalized_residuals.shape)
        * torch.finfo(singular_values.dtype).eps
        * largest_singular_value
    )
    effective_rank = int((singular_values > rank_tolerance).sum().item())
    if effective_rank == 0:
        raise ValueError(
            "The global residuals span a zero-rank subspace; cannot construct "
            "a unit residual direction"
        )
    basis_rank = min(rank, effective_rank)
    diagnostics = {
        "subspace_effective_rank": effective_rank,
        "subspace_requested_rank": int(rank),
        "subspace_basis_rank": basis_rank,
        "subspace_normalized_before_svd": True,
        "subspace_singular_values": singular_values[:5].tolist(),
        "subspace_zero_norm_residual_count": int((residual_norms <= eps).sum().item()),
    }
    return vh[:basis_rank], diagnostics


def _prepare_residual_subspace(
    target_embeddings,
    anchor_embeddings,
    top_k,
    eps=1e-8,
    rank=None,
    normalize_before_svd=False,
):
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
    if rank is not None and (
        isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0
    ):
        raise ValueError(f"Residual rank must be a positive integer, got {rank}")

    legacy_residuals = anchor_embeddings - target_embeddings
    flattened_targets = target_embeddings.reshape(pair_count, -1).float()
    flattened_anchors = anchor_embeddings.reshape(pair_count, -1).float()
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
    svd_residuals = selected_residuals
    if normalize_before_svd:
        selected_norms = torch.linalg.vector_norm(svd_residuals, dim=1)
        svd_residuals = svd_residuals / selected_norms.unsqueeze(1).clamp_min(eps)

    _, singular_values, vh = torch.linalg.svd(svd_residuals, full_matrices=False)
    largest_singular_value = singular_values.max()
    rank_tolerance = (
        max(svd_residuals.shape)
        * torch.finfo(singular_values.dtype).eps
        * largest_singular_value
    )
    effective_rank = int((singular_values > rank_tolerance).sum().item())
    if effective_rank == 0:
        raise ValueError(
            "The selected residuals span a zero-rank subspace; cannot construct "
            "a unit residual direction"
        )
    if rank is not None and rank > min(svd_residuals.shape):
        raise ValueError(
            f"Residual rank {rank} exceeds the maximum possible rank "
            f"{min(svd_residuals.shape)} for residual matrix shape "
            f"{tuple(svd_residuals.shape)}"
        )
    requested_basis_rank = effective_rank if rank is None else rank
    basis = vh[:min(requested_basis_rank, effective_rank)]

    diagnostics = {
        "subspace_requested_top_k": int(top_k),
        "subspace_selected_indices": selected_indices.tolist(),
        "subspace_selected_scores": mean_cosine_scores[selected_indices].tolist(),
        "subspace_effective_rank": effective_rank,
        "subspace_requested_rank": None if rank is None else int(rank),
        "subspace_normalized_before_svd": bool(normalize_before_svd),
        "subspace_singular_values": singular_values[:5].tolist(),
    }
    return {
        "output_dtype": target_embeddings.dtype,
        "legacy_shape": legacy_residuals.shape,
        "flattened_targets": flattened_targets,
        "flattened_anchors": flattened_anchors,
        "flattened_legacy": flattened_legacy,
        "legacy_norms": legacy_norms,
        "basis": basis,
        "diagnostics": diagnostics,
    }


def _build_norm_matched_directions(context, primary_vectors, primary_sign, eps):
    basis = context["basis"]
    flattened_legacy = context["flattened_legacy"]
    legacy_norms = context["legacy_norms"]

    projected_primary = _project_onto_basis(primary_vectors, basis)
    projected_primary_norms = torch.linalg.vector_norm(projected_primary, dim=1)
    projected_legacy = _project_onto_basis(flattened_legacy, basis)
    projected_legacy_norms = torch.linalg.vector_norm(projected_legacy, dim=1)

    unit_directions = torch.empty_like(primary_vectors)
    primary_projection_mask = projected_primary_norms > eps
    legacy_fallback_mask = ~primary_projection_mask & (projected_legacy_norms > eps)
    basis_fallback_mask = ~primary_projection_mask & ~legacy_fallback_mask

    unit_directions[primary_projection_mask] = (
        primary_sign
        * projected_primary[primary_projection_mask]
        / projected_primary_norms[primary_projection_mask].unsqueeze(1)
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
    diagnostics = {
        "subspace_max_norm_error": norm_errors.max().item(),
        "subspace_max_projection_error": subspace_errors.max().item(),
        "subspace_primary_projection_fallback_count": int(
            (~primary_projection_mask).sum().item()
        ),
        "subspace_legacy_fallback_count": int(legacy_fallback_mask.sum().item()),
        "subspace_basis_fallback_count": int(basis_fallback_mask.sum().item()),
    }
    return flattened_residuals, diagnostics


def build_smallest_cosine_subspace_residuals(
    target_embeddings,
    anchor_embeddings,
    top_k,
    eps=1e-8,
):
    """Build norm-matched residuals opposite each target's subspace projection."""
    context = _prepare_residual_subspace(
        target_embeddings,
        anchor_embeddings,
        top_k,
        eps,
    )
    flattened_targets = context["flattened_targets"]
    flattened_legacy = context["flattened_legacy"]
    flattened_residuals, direction_diagnostics = _build_norm_matched_directions(
        context,
        primary_vectors=flattened_targets,
        primary_sign=-1.0,
        eps=eps,
    )

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

    residuals = flattened_residuals.reshape(context["legacy_shape"]).to(
        context["output_dtype"]
    )
    diagnostics = dict(context["diagnostics"])
    diagnostics.update(direction_diagnostics)
    diagnostics.update({
        "subspace_legacy_cosine_mean": legacy_cosines.mean().item(),
        "subspace_new_cosine_mean": new_cosines.mean().item(),
        "subspace_target_projection_fallback_count": direction_diagnostics[
            "subspace_primary_projection_fallback_count"
        ],
    })
    return residuals, diagnostics


def build_negative_target_normalized_residual_subspace_residuals(
    target_embeddings,
    anchor_embeddings,
    rank,
    eps=1e-8,
):
    """Build negative-target residuals from a rank-k SVD of normalized residuals."""
    context = _prepare_residual_subspace(
        target_embeddings,
        anchor_embeddings,
        top_k=target_embeddings.shape[0],
        eps=eps,
        rank=rank,
        normalize_before_svd=True,
    )
    flattened_targets = context["flattened_targets"]
    flattened_legacy = context["flattened_legacy"]
    flattened_residuals, direction_diagnostics = _build_norm_matched_directions(
        context,
        primary_vectors=flattened_targets,
        primary_sign=-1.0,
        eps=eps,
    )

    residuals = flattened_residuals.reshape(context["legacy_shape"]).to(
        context["output_dtype"]
    )
    diagnostics = dict(context["diagnostics"])
    diagnostics.update(direction_diagnostics)
    diagnostics.update({
        "subspace_legacy_cosine_mean": _cosine_with_shift(
            flattened_targets, flattened_legacy, eps
        ).mean().item(),
        "subspace_new_cosine_mean": _cosine_with_shift(
            flattened_targets, flattened_residuals, eps
        ).mean().item(),
    })
    return residuals, diagnostics


def build_global_pairwise_residual_subspace_residuals(
    target_embeddings,
    anchor_embeddings,
    concept_embeddings,
    extra_anchor_embeddings,
    rank,
    eps=1e-8,
):
    """Build negative-target residuals from a global pairwise residual basis."""
    if target_embeddings.ndim < 2:
        raise ValueError("Target embeddings must include a batch dimension")
    if target_embeddings.shape != anchor_embeddings.shape:
        raise ValueError(
            "Target and anchor embedding shapes must match, got "
            f"{tuple(target_embeddings.shape)} and {tuple(anchor_embeddings.shape)}"
        )
    if target_embeddings.shape[1:] != concept_embeddings.shape[1:]:
        raise ValueError(
            "Target and global concept embedding shapes must match, got "
            f"{tuple(target_embeddings.shape[1:])} and "
            f"{tuple(concept_embeddings.shape[1:])}"
        )

    global_residuals = build_global_pairwise_residual_matrix(
        concept_embeddings,
        extra_anchor_embeddings,
    )
    basis, basis_diagnostics = _build_normalized_residual_basis(
        global_residuals,
        rank=rank,
        eps=eps,
    )

    legacy_residuals = anchor_embeddings - target_embeddings
    pair_count = target_embeddings.shape[0]
    flattened_targets = target_embeddings.reshape(pair_count, -1).float()
    flattened_legacy = legacy_residuals.reshape(pair_count, -1).float()
    context = {
        "basis": basis,
        "flattened_legacy": flattened_legacy,
        "legacy_norms": torch.linalg.vector_norm(flattened_legacy, dim=1),
    }
    flattened_residuals, direction_diagnostics = _build_norm_matched_directions(
        context,
        primary_vectors=flattened_targets,
        primary_sign=-1.0,
        eps=eps,
    )

    residuals = flattened_residuals.reshape(legacy_residuals.shape).to(
        target_embeddings.dtype
    )
    diagnostics = dict(basis_diagnostics)
    diagnostics.update(direction_diagnostics)
    diagnostics.update({
        "global_subspace_concept_count": int(concept_embeddings.shape[0]),
        "global_subspace_extra_anchor_count": int(extra_anchor_embeddings.shape[0]),
        "global_subspace_residual_count": int(global_residuals.shape[0]),
        "global_subspace_residual_shape": tuple(global_residuals.shape),
        "subspace_legacy_cosine_mean": _cosine_with_shift(
            flattened_targets, flattened_legacy, eps
        ).mean().item(),
        "subspace_new_cosine_mean": _cosine_with_shift(
            flattened_targets, flattened_residuals, eps
        ).mean().item(),
        "subspace_target_projection_fallback_count": direction_diagnostics[
            "subspace_primary_projection_fallback_count"
        ],
    })
    return residuals, diagnostics


def build_target_global_pairwise_residual_subspace_residuals(
    target_embeddings,
    anchor_embeddings,
    extra_anchor_embeddings,
    rank,
    eps=1e-8,
):
    """Build a global residual basis using only targets and fixed anchors."""
    residuals, diagnostics = build_global_pairwise_residual_subspace_residuals(
        target_embeddings,
        anchor_embeddings,
        concept_embeddings=target_embeddings,
        extra_anchor_embeddings=extra_anchor_embeddings,
        rank=rank,
        eps=eps,
    )
    diagnostics.update({
        "target_global_subspace_target_count": diagnostics[
            "global_subspace_concept_count"
        ],
        "target_global_subspace_extra_anchor_count": diagnostics[
            "global_subspace_extra_anchor_count"
        ],
        "target_global_subspace_residual_count": diagnostics[
            "global_subspace_residual_count"
        ],
        "target_global_subspace_residual_shape": diagnostics[
            "global_subspace_residual_shape"
        ],
    })
    return residuals, diagnostics


def build_largest_anchor_cosine_subspace_residuals(
    target_embeddings,
    anchor_embeddings,
    top_k,
    eps=1e-8,
):
    """Build norm-matched residuals along each anchor's subspace projection."""
    context = _prepare_residual_subspace(
        target_embeddings,
        anchor_embeddings,
        top_k,
        eps,
    )
    flattened_targets = context["flattened_targets"]
    flattened_anchors = context["flattened_anchors"]
    flattened_legacy = context["flattened_legacy"]
    flattened_residuals, direction_diagnostics = _build_norm_matched_directions(
        context,
        primary_vectors=flattened_anchors,
        primary_sign=1.0,
        eps=eps,
    )

    legacy_anchor_cosines = _cosine_between(
        flattened_anchors,
        flattened_targets + flattened_legacy,
        eps,
    )
    new_anchor_cosines = _cosine_between(
        flattened_anchors,
        flattened_targets + flattened_residuals,
        eps,
    )
    residuals = flattened_residuals.reshape(context["legacy_shape"]).to(
        context["output_dtype"]
    )
    diagnostics = dict(context["diagnostics"])
    diagnostics.update(direction_diagnostics)
    diagnostics.update({
        "subspace_legacy_anchor_cosine_mean": legacy_anchor_cosines.mean().item(),
        "subspace_new_anchor_cosine_mean": new_anchor_cosines.mean().item(),
        "subspace_anchor_projection_fallback_count": direction_diagnostics[
            "subspace_primary_projection_fallback_count"
        ],
    })
    return residuals, diagnostics
