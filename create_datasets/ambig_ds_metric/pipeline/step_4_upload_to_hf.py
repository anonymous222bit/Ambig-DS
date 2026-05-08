#!/usr/bin/env python3
"""Stage the metric-ambiguity benchmark dataset and upload to HuggingFace.

Usage:
    python step_3_upload_to_hf.py --benchmark-dir ./benchmark              # stage only (dry run)
    python step_3_upload_to_hf.py --benchmark-dir ./benchmark --upload     # stage + upload
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


REPO_ID = "anonymous222bit/Ambig-DS-M"

README_TEMPLATE = """\
---
license: mit
task_categories:
  - text-generation
language:
  - en
tags:
  - benchmark
  - ambiguity
  - ml-engineering
  - kaggle
  - metric-ambiguity
pretty_name: "Ambig-DS-M: Metric Ambiguity Benchmark for ML Engineering Agents"
size_categories:
  - n<1K
---

# Ambig-DS-M: Metric Ambiguity Benchmark

A benchmark for measuring how well ML engineering agents handle **ambiguous evaluation metrics** in Kaggle-style competitions.

## What is this?

Each task is a Kaggle competition from [MLE-bench](https://github.com/openai/mle-bench) (OpenAI, 2024).
For every task we provide two prompt variants:

| Variant | File | Description |
|---------|------|-------------|
| **Full** | `prompts/{{slug}}/full.md` | Original Kaggle competition description — includes the exact evaluation metric, formula, and optimization direction |
| **Ambiguous** | `prompts/{{slug}}/ambig_metric.md` | Same description with metric information redacted — the Evaluation section is replaced with *"Submissions are scored against the held-out ground truth on this task."* |

The agent receives one of these prompts, builds a model on the competition data, and produces a submission CSV.
The submission is graded against the **true** metric using MLE-bench's grading infrastructure.

## Dataset Contents

```
prompts/
  {{slug}}/
    full.md              # Full competition description (metric included)
    ambig_metric.md      # Metric-redacted version
task_list.txt            # {n_tasks} competition slugs
metric_manifest.json     # Structured metric metadata per task
metrics_classified.csv   # Metric classification and grader info
edits_log.md             # Documentation of exactly what was changed per task
```

## Setup: Getting the Competition Data

This dataset contains **prompts only** — not the Kaggle competition data (train/test CSVs, images, etc.).
To download the data, install MLE-bench and run:

```bash
pip install -e git+https://github.com/openai/mle-bench.git
mlebench prepare --list task_list.txt --data-dir ./cache
```

This requires a [Kaggle API key](https://www.kaggle.com/docs/api) and acceptance of each competition's rules.

## Metric Manifest

`metric_manifest.json` provides structured metadata for each task's true evaluation metric:

```json
{{
  "spooky-author-identification": {{
    "metric_name": "Multi-class log loss",
    "metric_description": "Multi-class logarithmic loss over the three author classes...",
    "submission_format": "id, EAP, HPL, MWS — one probability per author class per row.",
    "is_lower_better": true,
    "notes": "Sample submission rows are 0.33,0.33,0.33 (uniform)."
  }}
}}
```

This metadata is used in the **clarify** experimental condition, where an answerer LLM responds to the agent's clarifying questions about the metric.

## Task List ({n_tasks} competitions)

{task_table}

## How Prompts Were Redacted

The redaction follows a consistent protocol (documented in `edits_log.md`):

1. The **Evaluation section body** is replaced with: *"Submissions are scored against the held-out ground truth on this task."*
2. **Inline metric mentions** (names, formulas, optimization direction) are neutralized
3. **Submission format hints** like "predict a probability" are changed to "predict the outcome"
4. **Submission column names** are preserved (the grader requires them)
5. **Data descriptions** are kept verbatim

## Citation

```bibtex
@article{{ambig-ds-m-2026,
  title={{Ambig-DS-M: Measuring the Impact of Metric Ambiguity on ML Engineering Agents}},
  year={{2026}}
}}
```

## License

The prompt files are derivative works of publicly available Kaggle competition descriptions,
redistributed following the precedent set by [MLE-bench](https://github.com/openai/mle-bench) (MIT License).
The metadata and redacted prompts are our original contributions.

This license applies to the files in this repository.
The underlying Kaggle competition datasets must be downloaded separately and are subject
to each competition's individual rules and terms of use.
"""


def build_task_table(slugs: list[str], manifest: dict) -> str:
    lines = ["| # | Competition | Metric | Direction |",
             "|---|---|---|---|"]
    for i, slug in enumerate(slugs, 1):
        entry = manifest.get(slug, {})
        metric = entry.get("metric_name", "—")
        direction = "lower" if entry.get("is_lower_better", False) else "higher"
        lines.append(f"| {i} | `{slug}` | {metric} | {direction} |")
    return "\n".join(lines)


def load_verify_summary(benchmark_dir: Path) -> dict | None:
    """Read <bench>/_verify/_summary.json if present."""
    p = benchmark_dir / "_verify" / "_summary.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def load_verify_alternatives(benchmark_dir: Path, slugs: list[str]) -> dict[str, list[str]]:
    """Read per-slug judge files and pull out plausible alternatives."""
    out: dict[str, list[str]] = {}
    vdir = benchmark_dir / "_verify"
    if not vdir.exists():
        return out
    for slug in slugs:
        p = vdir / f"{slug}.json"
        if not p.exists():
            continue
        try:
            v = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        alts = (v.get("checks", {})
                 .get("plausible_alternatives", {})
                 .get("alternatives", []))
        if alts:
            out[slug] = list(alts)
    return out


def stage(benchmark_dir: Path, staging_dir: Path,
          allow_failed: bool = False, require_verify: bool = False) -> dict:
    """Copy benchmark files into a clean staging directory.

    Returns the verify summary (or {}) so the caller can print it.
    """
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True)

    prompts_dir = benchmark_dir / "prompts"
    task_list_path = benchmark_dir / "task_list.txt"
    manifest_path = prompts_dir / "_metric_manifest.json"
    if not manifest_path.exists():
        manifest_path = benchmark_dir / "metric_manifest.json"
    edits_log = benchmark_dir / "edits_log.md"
    if not edits_log.exists():
        edits_log = prompts_dir / "EDITS_LOG.md"

    slugs = task_list_path.read_text().strip().splitlines()
    manifest = json.loads(manifest_path.read_text())

    # Verifier gating (paper Section 3.3 checklist).
    summary = load_verify_summary(benchmark_dir) or {}
    verdicts = summary.get("verdicts", {})
    if require_verify and not verdicts:
        raise SystemExit(
            "--require-verify set but no _verify/_summary.json found. "
            "Run step_3_llm_verify.py first.")
    if verdicts and not allow_failed:
        failed = [s for s in slugs if verdicts.get(s) == "fail"]
        if failed:
            raise SystemExit(
                f"{len(failed)} slug(s) failed LLM verification: {failed}. "
                f"Re-run step_3_llm_verify.py, fix the prompts, or pass "
                f"--allow-failed to override.")

    # 1. task_list.txt
    shutil.copy2(task_list_path, staging_dir / "task_list.txt")

    # 2. metric_manifest.json (strip the _doc key; attach validated alternatives)
    clean_manifest = {k: v for k, v in manifest.items() if k != "_doc"}
    alternatives = load_verify_alternatives(benchmark_dir, slugs)
    for slug, alts in alternatives.items():
        if slug in clean_manifest:
            clean_manifest[slug]["validated_alternatives"] = alts
    (staging_dir / "metric_manifest.json").write_text(
        json.dumps(clean_manifest, indent=2, ensure_ascii=False) + "\n"
    )

    # 3. metrics_classified.csv (if it exists)
    for csv_name in ["mle_bench_metrics_classified.csv", "metrics_classified.csv"]:
        csv_path = benchmark_dir / csv_name
        if csv_path.exists():
            shutil.copy2(csv_path, staging_dir / "metrics_classified.csv")
            break

    # 4. edits_log.md
    if edits_log.exists():
        shutil.copy2(edits_log, staging_dir / "edits_log.md")

    # 5. prompts/{slug}/full.md and ambig_metric.md
    prompts_out = staging_dir / "prompts"
    for slug in slugs:
        slug_dir = prompts_out / slug
        slug_dir.mkdir(parents=True)
        shutil.copy2(prompts_dir / slug / "full.md", slug_dir / "full.md")
        shutil.copy2(prompts_dir / slug / "ambig_metric.md", slug_dir / "ambig_metric.md")

    # 6. README.md
    task_table = build_task_table(slugs, clean_manifest)
    readme = README_TEMPLATE.format(n_tasks=len(slugs), task_table=task_table)
    (staging_dir / "README.md").write_text(readme)

    # Summary
    n_files = sum(1 for _ in staging_dir.rglob("*") if _.is_file())
    total_bytes = sum(f.stat().st_size for f in staging_dir.rglob("*") if f.is_file())
    print(f"Staged {n_files} files ({total_bytes / 1024:.0f} KB) in {staging_dir}")
    print(f"  task_list.txt:        {len(slugs)} tasks")
    print(f"  metric_manifest.json: {len(clean_manifest)} entries "
          f"({sum(1 for s in clean_manifest.values() if 'validated_alternatives' in s)} with alternatives)")
    print(f"  prompts/:             {len(slugs)} dirs x 2 files")
    if summary:
        print(f"  verifier:             {summary.get('pass', 0)}/{summary.get('n', 0)} pass "
              f"(model={summary.get('model', '?')})")
    return summary


def upload(staging_dir: Path, repo_id: str):
    """Upload staged directory to HuggingFace."""
    from huggingface_hub import login, upload_folder

    login()
    print(f"\nUploading {staging_dir} to {repo_id}...")
    upload_folder(
        folder_path=str(staging_dir),
        repo_id=repo_id,
        repo_type="dataset",
    )
    print("Done.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark-dir", type=Path, required=True,
                    help="Benchmark directory with prompts/")
    ap.add_argument("--upload", action="store_true",
                    help="Upload to HuggingFace after staging (default: stage only)")
    ap.add_argument("--staging-dir", type=Path, default=None,
                    help="Staging directory (default: <benchmark-dir>/_hf_staging)")
    ap.add_argument("--repo-id", default=REPO_ID,
                    help=f"HuggingFace repo ID (default: {REPO_ID})")
    ap.add_argument("--allow-failed", action="store_true",
                    help="Stage even slugs that failed step_3_llm_verify.py.")
    ap.add_argument("--require-verify", action="store_true",
                    help="Refuse to stage if no verifier summary is present.")
    args = ap.parse_args()

    benchmark_dir = args.benchmark_dir.resolve()
    staging_dir = args.staging_dir or (benchmark_dir / "_hf_staging")

    stage(benchmark_dir, staging_dir,
          allow_failed=args.allow_failed,
          require_verify=args.require_verify)

    if args.upload:
        upload(staging_dir, args.repo_id)
    else:
        print(f"\nDry run. Inspect {staging_dir}/ then re-run with --upload")


if __name__ == "__main__":
    main()
