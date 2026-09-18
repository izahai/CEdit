"""Proxy redirecting to remote_scripts.eval_zero_anchor.eval_zero_anchor_img.preview_concat."""

from pathlib import Path
import sys

current = Path(__file__).resolve()
REPO_ROOT = None
for parent in current.parents:
    if (parent / "train_erase_null.py").exists() or (parent / "src").exists():
        REPO_ROOT = parent
        break
if REPO_ROOT is None:
    REPO_ROOT = current.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from remote_scripts.eval_zero_anchor.eval_zero_anchor_img.preview_concat import *

if __name__ == "__main__":
    cli_args = parse_args()
    run_pipeline(cli_args)
