import os, re, pdb
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import time
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from kmeans_pytorch import kmeans
from diffusers import StableDiffusionPipeline
from src.utils import seed_everything


def get_token_id(prompt, tokenizer=None, return_ids_only=True):
    token_ids = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    return token_ids.input_ids if return_ids_only else token_ids


def build_target_anchor_statistics(
    target_embeddings,
    anchor_embeddings,
    anchor_mode="legacy",
    residual_scale=1.0,
    residual_rank=1,
):
    if not target_embeddings or len(target_embeddings) != len(anchor_embeddings):
        raise ValueError("Target and anchor embeddings must be non-empty and have the same length")
    if not np.isfinite(residual_scale) or residual_scale <= 0:
        raise ValueError(f"Residual scale must be a positive finite value, got {residual_scale}")

    target_embeddings = torch.stack(target_embeddings)
    anchor_embeddings = torch.stack(anchor_embeddings)
    if target_embeddings.shape != anchor_embeddings.shape:
        raise ValueError(
            f"Target and anchor embedding shapes must match, got "
            f"{target_embeddings.shape} and {anchor_embeddings.shape}"
        )

    sum_target_target = torch.stack([
        target_embs.T @ target_embs for target_embs in target_embeddings
    ]).mean(0)
    selected_residual_index = None
    selected_residual_norm = None
    positive_sign_count = None
    negative_sign_count = None
    selected_medoid_index = None
    selected_medoid_norm = None
    selected_medoid_score = None
    selected_medoid_similarity = None
    shared_residual_target_index = None
    truncated_svd_requested_rank = None
    truncated_svd_explained_energy = None
    truncated_svd_relative_error = None

    if anchor_mode == "legacy":
        residuals = anchor_embeddings - target_embeddings
    elif anchor_mode == "shared_residual_mean":
        shared_residual_mean = anchor_embeddings.mean(0) - target_embeddings.mean(0)
        residuals = shared_residual_mean.unsqueeze(0).expand_as(target_embeddings)
    elif anchor_mode == "shared_residual_max_norm":
        candidate_residuals = anchor_embeddings - target_embeddings
        candidate_residual_norms = torch.linalg.vector_norm(
            candidate_residuals.reshape(candidate_residuals.shape[0], -1),
            dim=1,
        )
        selected_residual_index = candidate_residual_norms.argmax().item()
        shared_residual_target_index = selected_residual_index
        shared_residual_max_norm = candidate_residuals[selected_residual_index]
        residuals = shared_residual_max_norm.unsqueeze(0).expand_as(target_embeddings)
    elif anchor_mode in [
        "shared_residual_cosine_medoid",
        "shared_residual_abs_cosine_medoid",
        "shared_residual_smallest_cosine_medoid",
    ]:
        candidate_residuals = anchor_embeddings - target_embeddings
        flattened_residuals = candidate_residuals.reshape(
            candidate_residuals.shape[0], -1
        ).float()
        residual_norms = torch.linalg.vector_norm(flattened_residuals, dim=1, keepdim=True)
        normalized_residuals = flattened_residuals / residual_norms.clamp_min(
            torch.finfo(flattened_residuals.dtype).eps
        )
        cosine_similarity = normalized_residuals @ normalized_residuals.T
        if anchor_mode == "shared_residual_abs_cosine_medoid":
            cosine_similarity = cosine_similarity.abs()
            selected_medoid_similarity = "absolute cosine"
        elif anchor_mode == "shared_residual_smallest_cosine_medoid":
            selected_medoid_similarity = "smallest cosine"
        else:
            selected_medoid_similarity = "cosine"

        if candidate_residuals.shape[0] > 1:
            mean_similarity = (
                cosine_similarity.sum(dim=1) - cosine_similarity.diagonal()
            ) / (candidate_residuals.shape[0] - 1)
        else:
            mean_similarity = cosine_similarity.diagonal()

        if anchor_mode == "shared_residual_smallest_cosine_medoid":
            selected_medoid_index = mean_similarity.argmin().item()
        else:
            selected_medoid_index = mean_similarity.argmax().item()
        shared_residual_target_index = selected_medoid_index
        selected_medoid_score = mean_similarity[selected_medoid_index].item()
        shared_residual_medoid = candidate_residuals[selected_medoid_index]
        residuals = shared_residual_medoid.unsqueeze(0).expand_as(target_embeddings)
    elif anchor_mode == "shared_residual_sign_aligned":
        candidate_residuals = anchor_embeddings - target_embeddings
        shared_residual = candidate_residuals.mean(0)
        residual_dots = (
            candidate_residuals * shared_residual.unsqueeze(0)
        ).reshape(candidate_residuals.shape[0], -1).sum(dim=1)
        residual_signs = torch.where(
            residual_dots < 0,
            -torch.ones_like(residual_dots),
            torch.ones_like(residual_dots),
        )
        sign_shape = [residual_signs.shape[0]] + [1] * shared_residual.ndim
        residuals = residual_signs.reshape(sign_shape) * shared_residual.unsqueeze(0)
        positive_sign_count = (residual_signs > 0).sum().item()
        negative_sign_count = (residual_signs < 0).sum().item()
    elif anchor_mode == "truncated_svd_residual":
        if (
            not isinstance(residual_rank, (int, np.integer))
            or isinstance(residual_rank, (bool, np.bool_))
            or residual_rank <= 0
        ):
            raise ValueError(f"Residual rank must be a positive integer, got {residual_rank}")

        candidate_residuals = anchor_embeddings - target_embeddings
        flattened_residuals = candidate_residuals.reshape(
            candidate_residuals.shape[0], -1
        ).float()
        max_residual_rank = min(flattened_residuals.shape)
        if residual_rank > max_residual_rank:
            raise ValueError(
                f"Residual rank {residual_rank} exceeds the maximum possible rank "
                f"{max_residual_rank} for residual matrix shape "
                f"{tuple(flattened_residuals.shape)}"
            )

        u, singular_values, vh = torch.linalg.svd(
            flattened_residuals,
            full_matrices=False,
        )
        truncated_residuals = (
            u[:, :residual_rank] * singular_values[:residual_rank]
        ) @ vh[:residual_rank]
        residual_energy = singular_values.square().sum()
        retained_energy = singular_values[:residual_rank].square().sum()
        if residual_energy > 0:
            truncated_svd_explained_energy = (
                retained_energy / residual_energy
            ).item()
            truncated_svd_relative_error = (
                torch.linalg.vector_norm(flattened_residuals - truncated_residuals)
                / torch.linalg.vector_norm(flattened_residuals)
            ).item()
        else:
            truncated_svd_explained_energy = 1.0
            truncated_svd_relative_error = 0.0

        truncated_svd_requested_rank = int(residual_rank)
        residuals = truncated_residuals.reshape_as(candidate_residuals).to(
            candidate_residuals.dtype
        )
    else:
        raise ValueError(f"Invalid anchor mode: {anchor_mode}")

    residuals = residuals * residual_scale
    target_anchor_delta = torch.stack([
        residual_embs.T @ target_embs
        for residual_embs, target_embs in zip(residuals, target_embeddings)
    ]).mean(0)
    if selected_residual_index is not None:
        selected_residual_norm = torch.linalg.vector_norm(
            residuals[selected_residual_index].reshape(-1)
        ).item()
    if selected_medoid_index is not None:
        selected_medoid_norm = torch.linalg.vector_norm(
            residuals[selected_medoid_index].reshape(-1)
        ).item()

    residual_matrix = residuals.reshape(-1, residuals.shape[-1]).float()
    delta_singular_values = torch.linalg.svdvals(target_anchor_delta.float())
    max_singular_value = delta_singular_values.max()
    rank_tolerance = max(target_anchor_delta.shape) * torch.finfo(delta_singular_values.dtype).eps * max_singular_value
    diagnostics = {
        "max_residual_deviation": (residuals - residuals[[0]]).abs().max().item(),
        "residual_rank": torch.linalg.matrix_rank(residual_matrix).item(),
        "edit_statistic_rank": (delta_singular_values > rank_tolerance).sum().item(),
        "edit_statistic_singular_values": delta_singular_values[:5].tolist(),
        "selected_residual_index": selected_residual_index,
        "selected_residual_norm": selected_residual_norm,
        "residual_scale": residual_scale,
        "positive_sign_count": positive_sign_count,
        "negative_sign_count": negative_sign_count,
        "selected_medoid_index": selected_medoid_index,
        "selected_medoid_norm": selected_medoid_norm,
        "selected_medoid_score": selected_medoid_score,
        "selected_medoid_similarity": selected_medoid_similarity,
        "shared_residual_target_index": shared_residual_target_index,
        "truncated_svd_requested_rank": truncated_svd_requested_rank,
        "truncated_svd_explained_energy": truncated_svd_explained_energy,
        "truncated_svd_relative_error": truncated_svd_relative_error,
    }
    return sum_target_target, target_anchor_delta, diagnostics


def generate_perturbed_embs(ret_embs, P, erase_weight, num_per_sample, mini_batch=8):
    ret_embs = ret_embs.squeeze(1)
    out_embs, norm_list = [], []
    for i in range(0, ret_embs.size(0), mini_batch):
        mini_ret_embs = ret_embs[i:i + mini_batch]
        for _ in range(num_per_sample):
            noise = torch.randn_like(mini_ret_embs)
            perturbed_embs = mini_ret_embs + noise @ P
            out_embs.append(perturbed_embs)
            norm_list.append(torch.matmul(perturbed_embs, erase_weight.T).norm(dim=1))
    out_embs = torch.cat(out_embs, dim=0)
    norm_list = torch.cat(norm_list, dim=0)
    return out_embs[norm_list > norm_list.mean()].unsqueeze(1) # shape: [Num, 1, 768]


@torch.no_grad()
def edit_model(args, pipeline, target_concepts, anchor_concepts, retain_texts, baseline=None, chunk_size=128, emb_size=768, device="cuda"):

    I = torch.eye(emb_size, device=device)
    if args.params == 'KV':
        edit_dict = {k: v for k, v in pipeline.unet.state_dict().items() if 'attn2.to_k' in k or 'attn2.to_v' in k}
    elif args.params == 'V':
        edit_dict = {k: v for k, v in pipeline.unet.state_dict().items() if 'attn2.to_v' in k}
    elif args.params == 'K':
        edit_dict = {k: v for k, v in pipeline.unet.state_dict().items() if 'attn2.to_k' in k}

    if baseline in ['SPEED']:
        null_inputs = get_token_id('', pipeline.tokenizer, return_ids_only=False)
        null_hidden = pipeline.text_encoder(null_inputs.input_ids.to(device)).last_hidden_state[0]
        cluster_ids, cluster_centers = kmeans(X=null_hidden[1:], num_clusters=3, distance='euclidean', device='cuda')
        K2 = torch.cat([null_hidden[[0], :], cluster_centers.to(device)], dim=0).T
        I2 = torch.eye(len(K2.T), device=device)
    else:
        raise ValueError("Invalid baseline")

    # region [Target and Anchor]
    target_embeddings, anchor_embeddings = [], []
    for i in range(0, len(target_concepts)):
        target_inputs = get_token_id(target_concepts[i], pipeline.tokenizer, return_ids_only=False)
        target_embs = pipeline.text_encoder(target_inputs.input_ids.to(device)).last_hidden_state[0]
        anchor_inputs = get_token_id(anchor_concepts[i], pipeline.tokenizer, return_ids_only=False)
        anchor_embs = pipeline.text_encoder(anchor_inputs.input_ids.to(device)).last_hidden_state[0]
        if target_concepts == ['nudity']:
            target_embs = target_embs[1:, :]  # all tokens
            anchor_embs = anchor_embs[1:, :]  # all tokens
        else:
            target_embs = target_embs[[(target_inputs.attention_mask[0].sum().item() - 2)], :]  # last subject token
            anchor_embs = anchor_embs[[(anchor_inputs.attention_mask[0].sum().item() - 2)], :]  # last subject token
        target_embeddings.append(target_embs)
        anchor_embeddings.append(anchor_embs)
    anchor_mode = getattr(args, 'anchor_mode', 'legacy')
    sum_target_target, target_anchor_delta, anchor_diagnostics = build_target_anchor_statistics(
        target_embeddings,
        anchor_embeddings,
        anchor_mode=anchor_mode,
        residual_scale=getattr(args, 'residual_scale', 1.0),
        residual_rank=getattr(args, 'residual_rank', 1),
    )
    print(
        f"Anchor mode: {anchor_mode} | "
        f"residual scale: {anchor_diagnostics['residual_scale']:g} | "
        f"max residual deviation: {anchor_diagnostics['max_residual_deviation']:.3e} | "
        f"residual rank: {anchor_diagnostics['residual_rank']} | "
        f"edit statistic rank: {anchor_diagnostics['edit_statistic_rank']}"
    )
    print(f"Top edit statistic singular values: {anchor_diagnostics['edit_statistic_singular_values']}")
    if anchor_mode == "truncated_svd_residual":
        print("Residual source: truncated SVD of all target-anchor pairs")
    elif anchor_mode not in ["legacy", "shared_residual_mean"]:
        shared_target_index = anchor_diagnostics["shared_residual_target_index"]
        if shared_target_index is None:
            print("Shared residual source: mean residual (no single target prompt)")
        else:
            print(
                f"Shared residual source target: target[{shared_target_index}]="
                f"{target_concepts[shared_target_index]!r}"
            )
    if anchor_diagnostics["selected_residual_index"] is not None:
        print(
            f"Selected max-norm residual norm: "
            f"{anchor_diagnostics['selected_residual_norm']:.6f}"
        )
    if anchor_diagnostics["positive_sign_count"] is not None:
        print(
            f"Sign-aligned residual pairs: "
            f"+1.0={anchor_diagnostics['positive_sign_count']} | "
            f"-1.0={anchor_diagnostics['negative_sign_count']}"
        )
    if anchor_diagnostics["selected_medoid_index"] is not None:
        print(
            f"Selected {anchor_diagnostics['selected_medoid_similarity']} medoid residual: "
            f"mean similarity: {anchor_diagnostics['selected_medoid_score']:.6f} | "
            f"norm: {anchor_diagnostics['selected_medoid_norm']:.6f}"
        )
    if anchor_diagnostics["truncated_svd_requested_rank"] is not None:
        print(
            f"Truncated-SVD residual: requested rank="
            f"{anchor_diagnostics['truncated_svd_requested_rank']} | "
            f"effective rank={anchor_diagnostics['residual_rank']} | "
            f"explained energy="
            f"{anchor_diagnostics['truncated_svd_explained_energy']:.6f} | "
            f"relative reconstruction error="
            f"{anchor_diagnostics['truncated_svd_relative_error']:.6f}"
        )
    # endregion

    # region [Retain]
    last_ret_embs = []
    retain_texts = [text for text in retain_texts if not any(re.search(r'\b' + re.escape(concept.lower()) + r'\b', text.lower()) for concept in target_concepts)]
    assert len(retain_texts) + len(target_concepts) == len(set(retain_texts + target_concepts))
    for j in range(0, len(retain_texts), chunk_size):
        ret_inputs = get_token_id(retain_texts[j:j + chunk_size], pipeline.tokenizer, return_ids_only=False)
        ret_embs = pipeline.text_encoder(ret_inputs.input_ids.to(device)).last_hidden_state
        if retain_texts == ['']: 
            last_ret_embs.append(ret_embs[:, 1:, :].permute(1, 0, 2))
        else:
            last_subject_indices = ret_inputs.attention_mask.sum(1) - 2
            last_ret_embs.append(ret_embs[torch.arange(ret_embs.size(0)), last_subject_indices].unsqueeze(1))
    last_ret_embs = torch.cat(last_ret_embs)
    last_ret_embs = last_ret_embs[torch.randperm(last_ret_embs.size(0))]  # shuffle
    # endregion

    for (layer_name, layer_weight) in tqdm(edit_dict.items(), desc="Model Editing"):

        erase_weight = layer_weight @ target_anchor_delta @ (I + sum_target_target).inverse()
        (U0, S0, V0) = torch.svd(layer_weight)
        P0_min = V0[:, -1:] @ V0[:, -1:].T

        if args.aug_num > 0 and not args.disable_filter:
            weight_norm_init = torch.matmul(last_ret_embs.squeeze(1), erase_weight.T).norm(dim=1)
            layer_ret_embs = last_ret_embs[weight_norm_init > weight_norm_init.mean()]
        else:
            layer_ret_embs = last_ret_embs

        sum_ret_ret, valid_num = [], 0
        for j in range(0, len(layer_ret_embs), chunk_size):
            chunk_ret_embs = layer_ret_embs[j:j + chunk_size]
            if args.aug_num > 0:
                chunk_ret_embs = torch.cat(
                    [chunk_ret_embs, generate_perturbed_embs(chunk_ret_embs, P0_min, erase_weight, num_per_sample=args.aug_num)], dim=0
                )
            valid_num += chunk_ret_embs.shape[0]
            sum_ret_ret.append((chunk_ret_embs.transpose(1, 2) @ chunk_ret_embs).sum(0))
        sum_ret_ret = torch.stack(sum_ret_ret, dim=0).sum(0) / valid_num

        if baseline == 'SPEED':
            U, S, V = torch.svd(sum_ret_ret)
            P = U[:, S < args.threshold] @ U[:, S < args.threshold].T
            M = (sum_target_target @ P + args.retain_scale * I).inverse()
            delta_weight = layer_weight @ target_anchor_delta @ P @ (I - M @ K2 @ (K2.T @ P @ M @ K2 + args.lamb * I2).inverse() @ K2.T @ P) @ M

        # Save edited weights
        edit_dict[layer_name] = layer_weight + delta_weight

    print(f"Current model status: Edited {str(target_concepts)} into {str(anchor_concepts)}")
    return edit_dict


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # Base Config
    parser.add_argument('--sd_ckpt', help='base version for stable diffusion', type=str, default='CompVis/stable-diffusion-v1-4')
    parser.add_argument('--save_path', type=str, default=None)
    parser.add_argument('--file_name', type=str, default=None)
    parser.add_argument('--seed', type=int, default=0)
    # Erase Config
    parser.add_argument('--target_concepts', type=str, required=True)
    parser.add_argument('--anchor_concepts', type=str, required=True)
    parser.add_argument(
        '--anchor_mode',
        choices=[
            'legacy',
            'shared_residual_mean',
            'shared_residual_max_norm',
            'shared_residual_sign_aligned',
            'shared_residual_cosine_medoid',
            'shared_residual_abs_cosine_medoid',
            'shared_residual_smallest_cosine_medoid',
            'truncated_svd_residual',
        ],
        default='legacy',
        help='Strategy used to construct target-anchor residuals',
    )
    parser.add_argument('--retain_path', type=str, default=None)
    parser.add_argument('--heads', type=str, default=None)
    parser.add_argument('--baseline', type=str, default='SPEED')
    # Hyperparameters
    parser.add_argument('--params', type=str, default='V')
    parser.add_argument('--aug_num', type=int, default=10)
    parser.add_argument('--threshold', type=float, default=1e-1)
    parser.add_argument('--retain_scale', type=float, default=1.0)
    parser.add_argument(
        '--residual_scale',
        type=float,
        default=1.0,
        help='Scale target-to-anchor residuals before computing edit statistics',
    )
    parser.add_argument(
        '--residual_rank',
        type=int,
        default=1,
        help='Rank retained by truncated_svd_residual',
    )
    parser.add_argument('--lamb', type=float, default=0.0)
    parser.add_argument('--disable_filter', action='store_true', default=False)
    args = parser.parse_args()
    if not np.isfinite(args.residual_scale) or args.residual_scale <= 0:
        parser.error('--residual_scale must be a positive finite value')
    if args.residual_rank <= 0:
        parser.error('--residual_rank must be a positive integer')
    device = torch.device("cuda")
    seed_everything(args.seed)

    target_concepts = [con.strip() for con in args.target_concepts.split(',')]
    if (
        args.anchor_mode == 'truncated_svd_residual'
        and args.residual_rank > len(target_concepts)
    ):
        parser.error(
            f'--residual_rank cannot exceed the number of target concepts '
            f'({len(target_concepts)})'
        )
    anchor_concepts = args.anchor_concepts
    retain_path = args.retain_path
    
    file_suffix = "_".join(target_concepts[:5]) + f"_{len(target_concepts)}"  # The filename only displays the first 5 target concepts in multi-concept erasure
    anchor_concepts = [x.strip() for x in anchor_concepts.split(',')]
    if len(anchor_concepts) == 1:
        anchor_concepts = anchor_concepts * len(target_concepts)
        if anchor_concepts[0] == "":
            file_suffix += '-to_null'
        else:
            file_suffix += f'-to_{anchor_concepts[0]}'
    else:
        assert len(target_concepts) == len(anchor_concepts)
        file_suffix += f'-to_{anchor_concepts[0]}_etc'
    if args.anchor_mode == 'shared_residual_mean':
        file_suffix += '-shared_residual_mean'
    elif args.anchor_mode == 'shared_residual_max_norm':
        file_suffix += '-shared_residual_max_norm'
    elif args.anchor_mode == 'shared_residual_sign_aligned':
        file_suffix += '-shared_residual_sign_aligned'
    elif args.anchor_mode == 'shared_residual_cosine_medoid':
        file_suffix += '-shared_residual_cosine_medoid'
    elif args.anchor_mode == 'shared_residual_abs_cosine_medoid':
        file_suffix += '-shared_residual_abs_cosine_medoid'
    elif args.anchor_mode == 'shared_residual_smallest_cosine_medoid':
        file_suffix += '-shared_residual_smallest_cosine_medoid'
    elif args.anchor_mode == 'truncated_svd_residual':
        file_suffix += f'-truncated_svd_residual_rank_{args.residual_rank}'
    if args.residual_scale != 1.0:
        file_suffix += f'-residual_scale_{args.residual_scale:g}'

    retain_texts = []
    if retain_path is not None:
        assert retain_path.endswith('.csv')
        df = pd.read_csv(retain_path)
        for head in args.heads.split(','):
            retain_texts += df[head.strip()].unique().tolist()
    else:
        retain_texts.append("")

    pipeline = StableDiffusionPipeline.from_pretrained(args.sd_ckpt).to(device)

    edit_dict = edit_model(
        args=args,
        pipeline=pipeline, 
        target_concepts=target_concepts, 
        anchor_concepts=anchor_concepts, 
        retain_texts=retain_texts, 
        baseline=args.baseline, 
        device=device, 
    )

    save_path = args.save_path or "logs/checkpoints"
    file_name = args.file_name or f"{time.strftime('%Y%m%d-%H%M%S')}-{file_suffix}"
    os.makedirs(save_path, exist_ok=True)
    torch.save(edit_dict, os.path.join(save_path, f"{file_name}.pt"))
