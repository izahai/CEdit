#!/usr/bin/env python3
"""Summarize four A/B probabilities and render the two-image probe figure."""

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


def slug(value):
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()


def read_frame(gcd_root, state, identity):
    path = gcd_root / f"{state}_{slug(identity)}.csv"
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"Empty evaluation CSV: {path}")
    frame["source_csv"] = str(path)
    return frame


def closest_to_mean(frame):
    values = frame[["p_target_a", "p_anchor_b"]].to_numpy(float)
    center = values.mean(axis=0, keepdims=True)
    index = int(np.square(values - center).sum(axis=1).argmin())
    return frame.iloc[index]


def boolean_mean(series):
    if pd.api.types.is_bool_dtype(series):
        return float(series.mean())
    normalized = series.astype(str).str.strip().str.casefold()
    return float(normalized.isin({"true", "1", "yes"}).mean())


def render_figure(edit_frames, target_a, anchor_b, figure_root):
    figure, axes = plt.subplots(1, 2, figsize=(6.8, 3.75))
    for axis, identity in zip(axes, (target_a, anchor_b)):
        frame = edit_frames[identity]
        representative = closest_to_mean(frame)
        with Image.open(representative["image_path"]) as image:
            axis.imshow(image.convert("RGB"))
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_title(f'Prompt: "{identity}"', fontsize=10)
        axis.set_xlabel(
            f"P({target_a}) = {frame['p_target_a'].mean():.3f}\n"
            f"P({anchor_b}) = {frame['p_anchor_b'].mean():.3f}",
            fontsize=9,
        )
        for spine in axis.spines.values():
            spine.set_linewidth(0.8)
    figure.suptitle(
        f'Edited model: "{target_a}" $\\rightarrow$ "{anchor_b}"\n'
        f"Mean GCD probabilities over {len(next(iter(edit_frames.values())))} images",
        fontsize=10,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.93), w_pad=1.2)
    figure_root.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_root / "erase_A_anchor_B_probe.pdf", bbox_inches="tight")
    figure.savefig(figure_root / "erase_A_anchor_B_probe.png", dpi=600, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gcd-root", type=Path, required=True)
    parser.add_argument("--summary-root", type=Path, required=True)
    parser.add_argument("--figure-root", type=Path, required=True)
    parser.add_argument("--target-a", required=True)
    parser.add_argument("--anchor-b", required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    args = parser.parse_args()

    frames = {}
    summary_rows = []
    for state in ("original", "edit"):
        for identity in (args.target_a, args.anchor_b):
            frame = read_frame(args.gcd_root, state, identity)
            if len(frame) != args.expected_count:
                raise ValueError(
                    f"Expected {args.expected_count} rows for {state}/{identity}, "
                    f"found {len(frame)}"
                )
            frames[(state, identity)] = frame
            summary_rows.append({
                "model_state": state,
                "prompt_identity": identity,
                "image_count": len(frame),
                "face_detection_rate": boolean_mean(frame["face_detected"]),
                "mean_p_target_a": frame["p_target_a"].mean(),
                "mean_p_anchor_b": frame["p_anchor_b"].mean(),
            })

    args.summary_root.mkdir(parents=True, exist_ok=True)
    pd.concat(frames.values(), ignore_index=True).to_csv(
        args.summary_root / "per_image_scores.csv", index=False
    )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.summary_root / "summary.csv", index=False)
    edited_a = frames[("edit", args.target_a)]
    edited_b = frames[("edit", args.anchor_b)]
    four_points = [
        {"metric": "P(A|x_A)", "value": edited_a["p_target_a"].mean()},
        {"metric": "P(B|x_A)", "value": edited_a["p_anchor_b"].mean()},
        {"metric": "P(A|x_B)", "value": edited_b["p_target_a"].mean()},
        {"metric": "P(B|x_B)", "value": edited_b["p_anchor_b"].mean()},
    ]
    with (args.summary_root / "four_points.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerows(four_points)
    metadata = {
        "target_a": args.target_a,
        "anchor_b": args.anchor_b,
        "scores": {row["metric"]: row["value"] for row in four_points},
        "score_definition": "mean exact GCD ResNet-50 softmax probability; no-face images contribute zero",
    }
    (args.summary_root / "four_points.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    render_figure(
        {args.target_a: edited_a, args.anchor_b: edited_b},
        args.target_a,
        args.anchor_b,
        args.figure_root,
    )
    print(pd.DataFrame(four_points).to_string(index=False))


if __name__ == "__main__":
    main()
