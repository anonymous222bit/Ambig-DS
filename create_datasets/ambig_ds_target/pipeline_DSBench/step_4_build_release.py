"""Step 4 - BUILD the public HF-shaped release locally. No upload.

Reads the working tree produced by steps 1+2 (decoy CSVs + ambig prompts)
and assembles a publishable directory tree:

    <out_dir>/
        README.md
        tasks.csv               # flat index, one row per task
        tasks/<slug>/
            task.txt            # clean (non-ambiguous) task prompt
            task_ambig.txt      # target-ambiguous variant
            eval.py             # per-task DSBench-style evaluator
            eval.json           # (optional) sidecar for custom-metric tasks
            _manifest.json      # public schema: source/task/ambig_recipe/diagnostics

The actual data CSVs are intentionally NOT included (redistribution-safe).
Consumers re-derive the ambig CSVs from the recorded seed in `_manifest.json`
using `evaluate/ambig_ds_target/step_1_setup_benchmark.py --release-source local`.

Uploading is the job of the next step (`step_5_upload_to_hf.py`).

Usage:
    AMBIG_DSBENCH_ROOT=/path/to/workspace \\
        python step_4_build_release.py [--out ./release]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path

# Layout root containing Dataset/, DSBench/, final_data_v3/.
PROJ = Path(os.environ.get("AMBIG_DSBENCH_ROOT", Path.cwd())).resolve()
NEW_ROOT = Path(os.environ.get(
    "AMBIG_PIPELINE_ROOT", Path(__file__).resolve().parent))
HF_REPO_ID = os.environ.get("AMBIG_HF_REPO", "anonymous222bit/Ambig-DS-T")

# Authoritative source of "what tasks exist": the ambig data folder.
# 50 OLD (DSBench wave) + N NEW (kaggle_new_dataset wave).
AMBIG_DATA_DIR = (
    PROJ
    / "final_data_v3/target_ambig/data_modeling/data/data/data_ambig_target_v3_gen"
)
# Both waves drop their decoy manifest under final_data_v3/target_ambig/data/<slug>/.
DECOY_MANIFEST_DIR = PROJ / "final_data_v3/target_ambig/data"
# Full (non-ambig) prompts: same path for both waves (mirror_to_dsbench writes here).
FULL_PROMPT_DIR = PROJ / "Dataset/data_modeling/data/data/task"
AMBIG_PROMPT_DIR = (
    PROJ
    / "final_data_v3/target_ambig/data_modeling/data/data/task_ambig_target_v3_gen"
)
# Eval scripts. NEW wave: kaggle_new_dataset/evaluation/<slug>_eval.py.
# OLD wave (and any new tasks promoted into DSBench): DSBench/data_modeling/evaluation/.
NEW_EVAL_DIR = NEW_ROOT / "evaluation"
DSBENCH_EVAL_DIR = PROJ / "DSBench/data_modeling/evaluation"

DEFAULT_OUT = Path(__file__).resolve().parent / "release"
# Module-level sink that build_one_task() / write_index() write into.
# Set by build_release(); kept module-level for back-compat.
OUT: Path = DEFAULT_OUT


def discover_slugs() -> list[str]:
    return sorted(p.name for p in AMBIG_DATA_DIR.iterdir() if p.is_dir())


def detect_wave(slug: str) -> str:
    """NEW wave tasks were processed via run_pipeline.sh and have a competitions/<slug>/ dir."""
    if (NEW_ROOT / "competitions" / slug).is_dir():
        return "kaggle_2026"
    return "dsbench_original"


def kaggle_url(slug: str) -> str:
    """Best-effort canonical Kaggle URL. Most slugs map directly; a few legacy
    DSBench tasks (e.g. `tabular-playground-series-jan-2021`) map to the same
    URL pattern. We do not validate liveness here."""
    return f"https://www.kaggle.com/competitions/{slug}"


def build_public_manifest(slug: str, wave: str) -> dict:
    """Strip nothing, but reorganize for clarity. The plaintext feature_map is
    intentionally kept so that the GitHub build script can rebuild the ambig
    arm bit-identically. Anyone running the benchmark already has the data
    locally (they downloaded it from Kaggle), so the map gives no extra leak."""
    src = json.loads((DECOY_MANIFEST_DIR / slug / "_manifest.json").read_text())

    return {
        "schema_version": "1.0",
        "slug": slug,
        "source": {
            "platform": "kaggle",
            "url": kaggle_url(slug),
            "rules_url": f"{kaggle_url(slug)}/rules",
            "wave": wave,
        },
        "task": {
            "task_type": src["target_type"],
            "id_column": src["id_column"],
            "true_target_column_in_ambig": src["true_target_column"],
            "decoy_column_in_ambig": src["decoy_column"],
            "original_target_name": src["original_target_name"],
            "n_features": src["n_features"],
            "n_train": src["n_train"],
            "n_test": src["n_test"],
        },
        "ambig_recipe": {
            "method": src["decoy_method"],
            "feature_map": src["feature_map"],
            "anon_feature_columns": src["anon_feature_columns"],
            "decoy_pool_anon_features": src["decoy_pool_anon_features"],
            "decoy_pool_abs_spearman_with_truth": src[
                "decoy_pool_abs_spearman_with_truth"
            ],
            "noise_classification_frac": src["noise_classification_frac"],
            "noise_regression_sigma_frac": src["noise_regression_sigma_frac"],
            "seeds": src["seeds"],
        },
        "diagnostics": src["diagnostics"],
    }


def find_eval_script(slug: str, wave: str) -> Path:
    """Return the per-task evaluator. The CLI is the same for both waves
    (--answer_file --predict_file --path --name); the evaluator writes a
    single float to <path>/<name>/result.txt. The NEW wave's evaluator we
    wrote ourselves; OLD wave evaluators come from DSBench upstream."""
    if wave == "kaggle_2026":
        # NEW-wave: usually under our generated evaluation dir; if a task was
        # also mirrored into DSBench, prefer ours (it's the latest).
        cand = NEW_EVAL_DIR / f"{slug}_eval.py"
        if cand.exists():
            return cand
    cand = DSBENCH_EVAL_DIR / f"{slug}_eval.py"
    if cand.exists():
        return cand
    # Last-resort fallback: NEW-wave dir even if wave was misclassified.
    cand = NEW_EVAL_DIR / f"{slug}_eval.py"
    if cand.exists():
        return cand
    raise FileNotFoundError(
        f"missing eval script for {slug} "
        f"(checked {NEW_EVAL_DIR} and {DSBENCH_EVAL_DIR})")


def find_eval_sidecar(eval_script: Path) -> Path | None:
    """Some tasks have a sidecar JSON (e.g. {'higher_is_better': true}) for
    custom metrics that the auto-template detector cannot identify. Optional."""
    sidecar = eval_script.with_suffix(".json")
    return sidecar if sidecar.exists() else None


def build_one_task(slug: str) -> dict:
    wave = detect_wave(slug)
    out_dir = OUT / "tasks" / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    full_prompt = FULL_PROMPT_DIR / f"{slug}.txt"
    ambig_prompt = AMBIG_PROMPT_DIR / f"{slug}.txt"
    if not full_prompt.exists():
        raise FileNotFoundError(f"missing full prompt: {full_prompt}")
    if not ambig_prompt.exists():
        raise FileNotFoundError(f"missing ambig prompt: {ambig_prompt}")

    shutil.copy2(full_prompt, out_dir / "task.txt")
    shutil.copy2(ambig_prompt, out_dir / "task_ambig.txt")

    eval_script = find_eval_script(slug, wave)
    shutil.copy2(eval_script, out_dir / "eval.py")
    sidecar = find_eval_sidecar(eval_script)
    if sidecar is not None:
        shutil.copy2(sidecar, out_dir / "eval.json")

    manifest = build_public_manifest(slug, wave)
    manifest["eval"] = {
        "script": "eval.py",
        "cli": ("python eval.py --answer_file <answers.csv> "
                "--predict_file <submission.csv> --path <out_dir> --name <slug>"),
        "result_path": "<out_dir>/<slug>/result.txt",
        "sidecar_present": sidecar is not None,
    }
    (out_dir / "_manifest.json").write_text(json.dumps(manifest, indent=2))

    return {
        "slug": slug,
        "wave": wave,
        "url": manifest["source"]["url"],
        "task_type": manifest["task"]["task_type"],
        "n_features": manifest["task"]["n_features"],
        "n_train": manifest["task"]["n_train"],
        "n_test": manifest["task"]["n_test"],
        "true_target_column_in_ambig": manifest["task"][
            "true_target_column_in_ambig"
        ],
        "cv_true": manifest["diagnostics"].get("cv_true"),
        "cv_decoy": manifest["diagnostics"].get("cv_decoy"),
        "cv_ratio_decoy_over_true": manifest["diagnostics"].get(
            "cv_ratio_decoy_over_true"
        ),
        "correlation_truth_vs_decoy": manifest["diagnostics"].get(
            "correlation_truth_vs_decoy"
        ),
    }


README = """\
---
pretty_name: "Ambig-DS-T (Target-Ambiguous Data Science Benchmark)"
license: cc-by-4.0
language:
  - en
size_categories:
  - n<1K
task_categories:
  - tabular-classification
  - tabular-regression
tags:
  - data-science
  - benchmark
  - ambiguity
  - kaggle
  - llm-agents
configs:
  - config_name: tasks_index
    data_files:
      - split: train
        path: tasks.csv
    default: true
---

# Ambig-DS-T: Target-Ambiguous Data Science Benchmark (prompts only)

This repository contains the **task prompts** and **decoy recipes** for a
benchmark of {N} Kaggle data-science tasks, each in two variants:

- `task.txt` — the original (non-ambiguous) task description.
- `task_ambig.txt` — a *target-ambiguous* rewrite. Feature names are anonymized
  to `f_01, f_02, …`, the original target column name is hidden, and the
  training data exposes **two** candidate target columns `val_1` and `val_2`.
  Exactly one is the real target; the other is a feature-predictable decoy
  with the same marginal distribution. Picking the wrong one is the failure
  mode the benchmark measures.

## What is **not** redistributed here

To respect each Kaggle competition's terms of use, this dataset deliberately
contains **no CSV data**. To run the benchmark you must:

1. Accept each competition's rules in your browser (the `rules_url` field of
   `_manifest.json` links straight there).
2. Download the data with the official Kaggle CLI.
3. Use the build script (published separately on GitHub) to apply the
   deterministic decoy generation recipe recorded in `_manifest.json`. This
   reproduces the ambig arm bit-identically using the seeds we used.

## Layout

    tasks.csv                  # flat index of all {N} tasks
    tasks/
      <slug>/
        task.txt               # full / clean task prompt
        task_ambig.txt         # target-ambiguous variant
        eval.py                # per-task evaluator (DSBench CLI)
        eval.json              # (optional) sidecar for custom-metric tasks
        _manifest.json         # provenance + decoy recipe + diagnostics

`_manifest.json.ambig_recipe` contains the seed and CLI arguments used by the
decoy generator, plus the anonymized→original feature map needed to rebuild
the ambig CSVs from the user's locally-downloaded Kaggle data.

## Evaluating a submission

Every `eval.py` accepts the same CLI:

    python eval.py --answer_file data/test_answer.csv \\
                   --predict_file my_submission.csv \\
                   --path out --name <slug>

…and writes a single float to `out/<slug>/result.txt`.

## Provenance

- Two waves are present: `dsbench_original` (50 tasks, derived from DSBench's
  task list) and `kaggle_2026` (11 newer Kaggle Playground Series tasks
  scraped in 2026).
- All tasks share the same prompt and manifest schema; the `wave` field in
  `_manifest.json.source` records which.

## Notes on prompts

The clean prompts (`task.txt`) are short factual paraphrases of each
competition's overview and data-tab text, generated by an LLM cleaner. The
ambig prompts (`task_ambig.txt`) are derived rewrites that disclose the
two-target convention, anonymize feature references, and remove leakage of
the original target name.
"""


def write_index(rows: list[dict]) -> None:
    cols = list(rows[0].keys())
    with (OUT / "tasks.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def build_release(out_dir: Path = DEFAULT_OUT, slugs: list[str] | None = None) -> Path:
    """Build the public release directory at `out_dir`. Returns out_dir.

    Idempotent: wipes and recreates `out_dir`.
    """
    global OUT
    OUT = Path(out_dir).resolve()
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    if slugs is None:
        slugs = discover_slugs()
    print(f"[release] discovered {len(slugs)} tasks")

    rows = []
    for slug in slugs:
        try:
            rows.append(build_one_task(slug))
            print(f"  + {slug}")
        except Exception as e:
            print(f"  ! SKIP {slug}: {e}")

    rows.sort(key=lambda r: r["slug"])
    write_index(rows)
    (OUT / "README.md").write_text(README.format(N=len(rows)))

    n_old = sum(1 for r in rows if r["wave"] == "dsbench_original")
    n_new = sum(1 for r in rows if r["wave"] == "kaggle_2026")
    print(
        f"[release] wrote {len(rows)} tasks  ({n_old} dsbench_original, "
        f"{n_new} kaggle_2026)  ->  {OUT}"
    )
    return OUT


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT,
                   help=f"Output directory (default: {DEFAULT_OUT})")
    p.add_argument("--tasks", default="",
                   help="Comma-separated subset of slugs (default: all discovered)")
    args = p.parse_args()
    slugs = None
    if args.tasks:
        slugs = [s.strip() for s in args.tasks.split(",") if s.strip()]
    build_release(args.out, slugs=slugs)


if __name__ == "__main__":
    main()
