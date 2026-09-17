#!/usr/bin/env bash
set -Eeuo pipefail

WORKFLOW_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

for stage in 03_train.sh 04_infer.sh 05_eval.sh; do
    printf '\nRunning %s for 50_celebrity\n' "${stage}"
    BENCHMARK_NAMES_RAW=50_celebrity bash "${WORKFLOW_DIR}/${stage}"
done

printf '\nRebuilding the summary for all configured benchmarks\n'
bash "${WORKFLOW_DIR}/06_summarize.sh"

printf '\n50-celebrity negative-vs-positive ablation complete.\n'
