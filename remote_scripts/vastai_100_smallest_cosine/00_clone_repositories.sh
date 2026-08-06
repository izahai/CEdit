#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
load_workflow_config

require_directory "${REPO_ROOT}/.git"
if [[ -n "$(git -C "${REPO_ROOT}" status --porcelain)" ]]; then
    printf 'CEdit worktree is dirty; commit or stash changes before switching branches.\n' >&2
    exit 1
fi

git -C "${REPO_ROOT}" fetch origin "${CEDIT_BRANCH}"
git -C "${REPO_ROOT}" checkout "${CEDIT_BRANCH}"
git -C "${REPO_ROOT}" merge --ff-only "origin/${CEDIT_BRANCH}"

if [[ ! -d "${CE_EVAL_ROOT}/.git" ]]; then
    git clone "${CE_EVAL_REPOSITORY}" "${CE_EVAL_ROOT}"
fi
git -C "${CE_EVAL_ROOT}" fetch origin "${CE_EVAL_BRANCH}"
git -C "${CE_EVAL_ROOT}" checkout "${CE_EVAL_BRANCH}"
git -C "${CE_EVAL_ROOT}" merge --ff-only "origin/${CE_EVAL_BRANCH}"

printf 'CEdit:  %s @ %s\n' \
    "$(git -C "${REPO_ROOT}" branch --show-current)" \
    "$(git -C "${REPO_ROOT}" rev-parse --short HEAD)"
printf 'CE-Eval: %s @ %s\n' \
    "$(git -C "${CE_EVAL_ROOT}" branch --show-current)" \
    "$(git -C "${CE_EVAL_ROOT}" rev-parse --short HEAD)"
