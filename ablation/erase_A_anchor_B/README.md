# Erase celebrity A with celebrity B as anchor

This small ablation edits one celebrity target (A) toward a different
celebrity anchor (B):

\[
A\rightarrow B.
\]

It then generates matched images from prompts for both identities and evaluates
every image with the CE-Eval GCD pipeline. GCD first detects/crops a face and
then uses an adapted ResNet-50 with a 256-dimensional bottleneck and center-loss
celebrity classifier. Unlike the standard
top-5 evaluator, this workflow requests the complete softmax distribution so
that it can report the exact probabilities assigned to both A and B.

## Four primary scores

For the edited model, the workflow reports:

\[
P(A\mid x_A),\quad P(B\mid x_A),\quad
P(A\mid x_B),\quad P(B\mid x_B).
\]

These distinguish target erasure from identity replacement:

- low (P(A\mid x_A)) and high (P(B\mid x_A)): A is replaced by B;
- low (P(A\mid x_A)) while (P(B\mid x_B)) remains high: only A is erased;
- low (P(A\mid x_A)) and low (P(B\mid x_B)): evidence of joint erasure;
- high (P(A\mid x_B)): reverse identity contamination.

The original SD v1.4 scores and face-detection rates are also saved as controls.
Images with no detected face contribute probability zero and are separately
visible through `face_detection_rate`.

## Default experiment

`workflow.yaml` defaults to:

- A: `Adam Driver`;
- B: `Adriana Lima`;
- five standard celebrity prompt templates;
- four samples per template;
- 20 images per identity per model state;
- original and edited SD v1.4 with exactly matched latent noise;
- 50 diffusion steps and guidance scale 7.5.

The checkpoint uses the legacy directed residual (r=B-A), SPEED with
`params=V`, and an unrelated-celebrity retain set constructed from
`data/100_celebrity.csv` after excluding A and B.

Change `target_a` and `anchor_b` in `workflow.yaml` to test another pair. Both
names must appear in the benchmark and in the GCD label set.

## Run on a fresh Vast AI server

From the repository root:

```bash
cd /workspace/CEdit
tmux new -s erase-a-anchor-b
mkdir -p ablation/erase_A_anchor_B/outputs
bash ablation/erase_A_anchor_B/run_all.sh 2>&1 | \
  tee ablation/erase_A_anchor_B/outputs/run.log
```

Detach from tmux with `Ctrl-b d` and resume with:

```bash
tmux attach -t erase-a-anchor-b
```

Stages can be run individually:

```bash
bash ablation/erase_A_anchor_B/00_prepare.sh
bash ablation/erase_A_anchor_B/01_setup_environment.sh
bash ablation/erase_A_anchor_B/02_setup_ce_eval.sh
bash ablation/erase_A_anchor_B/03_train.sh
bash ablation/erase_A_anchor_B/04_generate.sh
bash ablation/erase_A_anchor_B/05_evaluate.sh
bash ablation/erase_A_anchor_B/06_summarize.sh
```

Resume is automatic. To rebuild a stage:

```bash
FORCE_RETRAIN=1 bash ablation/erase_A_anchor_B/03_train.sh
FORCE_RESAMPLE=1 bash ablation/erase_A_anchor_B/04_generate.sh
FORCE_EVAL=1 bash ablation/erase_A_anchor_B/05_evaluate.sh
FORCE_SUMMARY=1 bash ablation/erase_A_anchor_B/06_summarize.sh
```

## Outputs

```text
ablation/erase_A_anchor_B/outputs/
├── checkpoint/weight.pt
├── config/retain_concepts.csv
├── images/<a>_to_<b>/{A,B}/{original,edit}/*.png
├── gcd/{original,edit}_{a,b}.csv
├── summary/
│   ├── summary.csv
│   ├── per_image_scores.csv
│   ├── four_points.csv
│   └── four_points.json
└── figures/
    ├── erase_A_anchor_B_probe.pdf
    └── erase_A_anchor_B_probe.png
```

`summary/four_points.csv` is the compact numerical result. The figure shows one
representative edited image for prompt A and one for prompt B; the printed
probabilities are means over all generated images for that prompt.

## Download results

```bash
rsync -avz -e "ssh -p <PORT>" \
  root@<HOST>:/workspace/CEdit/ablation/erase_A_anchor_B/outputs/ \
  ablation/erase_A_anchor_B/outputs/
```
