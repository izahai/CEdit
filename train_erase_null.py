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


def build_target_anchor_statistics(target_embeddings, anchor_embeddings, anchor_mode="legacy"):
    if not target_embeddings or len(target_embeddings) != len(anchor_embeddings):
        raise ValueError("Target and anchor embeddings must be non-empty and have the same length")

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

    if anchor_mode == "legacy":
        sum_anchor_target = torch.stack([
            anchor_embs.T @ target_embs
            for target_embs, anchor_embs in zip(target_embeddings, anchor_embeddings)
        ]).mean(0)
        target_anchor_delta = sum_anchor_target - sum_target_target
        residuals = anchor_embeddings - target_embeddings
    elif anchor_mode == "shared_residual":
        shared_residual = anchor_embeddings.mean(0) - target_embeddings.mean(0)
        residuals = shared_residual.unsqueeze(0).expand_as(target_embeddings)
        target_anchor_delta = shared_residual.T @ target_embeddings.mean(0)
    else:
        raise ValueError(f"Invalid anchor mode: {anchor_mode}")

    residual_matrix = residuals.reshape(-1, residuals.shape[-1]).float()
    delta_singular_values = torch.linalg.svdvals(target_anchor_delta.float())
    max_singular_value = delta_singular_values.max()
    rank_tolerance = max(target_anchor_delta.shape) * torch.finfo(delta_singular_values.dtype).eps * max_singular_value
    diagnostics = {
        "max_residual_deviation": (residuals - residuals[[0]]).abs().max().item(),
        "residual_rank": torch.linalg.matrix_rank(residual_matrix).item(),
        "edit_statistic_rank": (delta_singular_values > rank_tolerance).sum().item(),
        "edit_statistic_singular_values": delta_singular_values[:5].tolist(),
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
    )
    print(
        f"Anchor mode: {anchor_mode} | "
        f"max residual deviation: {anchor_diagnostics['max_residual_deviation']:.3e} | "
        f"residual rank: {anchor_diagnostics['residual_rank']} | "
        f"edit statistic rank: {anchor_diagnostics['edit_statistic_rank']}"
    )
    print(f"Top edit statistic singular values: {anchor_diagnostics['edit_statistic_singular_values']}")
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
        choices=['legacy', 'shared_residual'],
        default='legacy',
        help='Use prompt anchors directly or construct virtual anchors with one shared residual',
    )
    parser.add_argument('--retain_path', type=str, default=None)
    parser.add_argument('--heads', type=str, default=None)
    parser.add_argument('--baseline', type=str, default='SPEED')
    # Hyperparameters
    parser.add_argument('--params', type=str, default='V')
    parser.add_argument('--aug_num', type=int, default=10)
    parser.add_argument('--threshold', type=float, default=1e-1)
    parser.add_argument('--retain_scale', type=float, default=1.0)
    parser.add_argument('--lamb', type=float, default=0.0)
    parser.add_argument('--disable_filter', action='store_true', default=False)
    args = parser.parse_args()
    device = torch.device("cuda")
    seed_everything(args.seed)

    target_concepts = [con.strip() for con in args.target_concepts.split(',')]
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
    if args.anchor_mode == 'shared_residual':
        file_suffix += '-shared_residual'

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
