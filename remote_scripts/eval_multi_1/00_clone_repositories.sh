#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
load_workflow_config

if [[ ! -d "${REPO_ROOT}/.git" ]]; then
    printf 'Expected an existing CEdit checkout at %s\n' "${REPO_ROOT}" >&2
    exit 1
fi

if [[ ! -d "${CE_EVAL_ROOT}/.git" ]]; then
    git clone --branch "${CE_EVAL_BRANCH}" "${CE_EVAL_REPOSITORY}" "${CE_EVAL_ROOT}"
else
    printf 'CE-Eval checkout already exists: %s\n' "${CE_EVAL_ROOT}"
fi

printf 'Repository setup complete.\n'
