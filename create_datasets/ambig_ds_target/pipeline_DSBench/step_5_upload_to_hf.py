"""Step 5 — UPLOAD the release/ directory to HuggingFace.

Thin wrapper around `step_4_build_release`. By default, rebuilds the release
locally first to guarantee on-disk parity with what gets uploaded.

Usage:
    AMBIG_DSBENCH_ROOT=/path/to/workspace \\
    AMBIG_HF_REPO=your-handle/Ambig-DS-T \\
    python step_5_upload_to_hf.py [--out ./release] [--skip-build]
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from step_4_build_release import DEFAULT_OUT, HF_REPO_ID, build_release


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT,
                   help=f"Release directory (default: {DEFAULT_OUT})")
    p.add_argument("--skip-build", action="store_true",
                   help="Use the existing release/ contents instead of rebuilding.")
    p.add_argument("--repo-id", default=HF_REPO_ID,
                   help=f"HF dataset repo id (default: {HF_REPO_ID})")
    args = p.parse_args()

    if args.skip_build:
        out = Path(args.out).resolve()
        if not (out / "tasks").exists():
            raise SystemExit(f"--skip-build but no release found at {out}")
        print(f"[upload] using existing release at {out}")
    else:
        out = build_release(args.out)

    from huggingface_hub import login, upload_folder

    login()
    upload_folder(
        folder_path=str(out),
        repo_id=args.repo_id,
        repo_type="dataset",
    )
    print(f"[upload] uploaded -> https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
