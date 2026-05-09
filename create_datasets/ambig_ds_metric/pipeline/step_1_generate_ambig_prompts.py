#!/usr/bin/env python3
"""Generate ambig_metric.md prompts from full.md for MLE-bench tasks.

For each slug that has prompts/{slug}/full.md but no ambig_metric.md,
calls an LLM to redact metric information following our redaction protocol.

Also generates _metric_manifest.json entries for new tasks.

Requires:
    OPENAI_API_KEY       (or .env file beside this script)
    OPENAI_BASE_URL      (optional; defaults to https://api.openai.com/v1)

Usage:
    python step_1_generate_ambig_prompts.py --benchmark-dir ./benchmark                # dry-run
    python step_1_generate_ambig_prompts.py --benchmark-dir ./benchmark --run          # generate all
    python step_1_generate_ambig_prompts.py --benchmark-dir ./benchmark --run --slugs foo bar
    python step_1_generate_ambig_prompts.py --benchmark-dir ./benchmark --run --manifest-only
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from _llm_client import call_llm, make_client, DEFAULT_MODEL

# ──────────────────────────────────────────────────────────────────────────────
# LLM prompts
# ──────────────────────────────────────────────────────────────────────────────

REDACTION_SYSTEM = """\
You are an expert editor for a research experiment on metric ambiguity in ML competitions.

Your job: given a Kaggle competition description (full.md), produce a REDACTED version
(ambig_metric.md) that removes all information about the evaluation metric while keeping
everything else identical.

RULES — follow these EXACTLY:

1. EVALUATION SECTION: Find the Evaluation / Metric section (usually under ## Evaluation
   or ### Evaluation). Replace its BODY with exactly this single sentence:
   "Submissions are scored against the held-out ground truth on this task."
   Keep the section header itself so the prompt structure is identical.

2. INLINE METRIC MENTIONS: Search the ENTIRE document for any other mentions of the
   specific metric (name, formula, optimization direction like "minimize"/"maximize",
   mathematical notation). Neutralize them:
   - If it says "predict a probability", change to "predict the outcome"
   - If it says "maximize AUC", remove the phrase
   - If it mentions a specific loss function, remove it
   - Do NOT remove submission column names (the grader needs them)
   - Do NOT remove data column descriptions

3. SUBMISSION FORMAT SECTION: Keep this section but neutralize metric hints:
   - Change "predict a probability for X" → "predict X"
   - Change "submit a real-valued score" → "submit a prediction"
   - Keep example CSV headers and column names intact
   - Keep format requirements (e.g., "must have a header")

4. PRESERVE EXACTLY:
   - All data descriptions, dataset structure, column definitions
   - All submission column names and CSV format examples
   - All markdown formatting, headers, links, images
   - Timeline, prizes, citation sections
   - Any text NOT related to the evaluation metric

5. DO NOT:
   - Add any new text or commentary
   - Change the document structure
   - Remove non-metric sections
   - Add notes about what was redacted

Output ONLY the redacted markdown. No preamble, no explanation."""

REDACTION_USER = """\
Here is the full Kaggle competition description (full.md). Produce the redacted version.

<full_md>
{full_md}
</full_md>"""

MANIFEST_SYSTEM = """\
You are extracting structured evaluation metric metadata from a Kaggle competition description.

Given a competition description, extract:
1. metric_name: Short name (e.g., "Multi-class log loss", "AUC (ROC)", "MCRMSE", "MAP@5")
2. metric_description: 1-3 sentence description of how the metric works
3. submission_format: Expected CSV columns and format (e.g., "id, target — probability in [0,1]")
4. is_lower_better: boolean — true if lower scores are better (losses), false if higher is better
5. notes: Any quirks about the metric (optional, can be empty string)

Output valid JSON only, no markdown fences."""

MANIFEST_USER = """\
Competition slug: {slug}

Description:
{full_md}

Additional context from grading code:
- Grader metric name: {grader_name}
- Eval excerpt: {eval_excerpt}"""


# ──────────────────────────────────────────────────────────────────────────────
# Utils
# ──────────────────────────────────────────────────────────────────────────────

def load_metrics_csv(benchmark_dir: Path) -> dict[str, dict]:
    """Load mle_bench_metrics_raw.csv into {slug: {grader_name, eval_excerpt, ...}}."""
    csv_path = benchmark_dir / "mle_bench_metrics_raw.csv"
    if not csv_path.exists():
        csv_path = benchmark_dir / "metrics_classified.csv"
    if not csv_path.exists():
        return {}
    out = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out[row["slug"]] = row
    return out


def load_manifest(benchmark_dir: Path) -> dict:
    manifest_path = benchmark_dir / "metric_manifest.json"
    if not manifest_path.exists():
        manifest_path = benchmark_dir / "prompts" / "_metric_manifest.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text())
    return {}


def save_manifest(manifest: dict, benchmark_dir: Path) -> None:
    manifest_path = benchmark_dir / "metric_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=4, ensure_ascii=False) + "\n")


def strip_code_fences(text: str) -> str:
    """Remove markdown code fences from LLM output."""
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    return text


def find_slugs_needing_ambig(prompts_dir: Path,
                              specific_slugs: list[str] | None = None) -> list[str]:
    """Find slugs that have full.md but no ambig_metric.md."""
    needed = []
    for slug_dir in sorted(prompts_dir.iterdir()):
        if not slug_dir.is_dir():
            continue
        slug = slug_dir.name
        if specific_slugs and slug not in specific_slugs:
            continue
        full_md = slug_dir / "full.md"
        ambig_md = slug_dir / "ambig_metric.md"
        if full_md.exists() and not ambig_md.exists():
            needed.append(slug)
    return needed


def find_slugs_needing_manifest(manifest: dict, prompts_dir: Path,
                                specific_slugs: list[str] | None = None) -> list[str]:
    """Find slugs that have full.md but no manifest entry."""
    needed = []
    for slug_dir in sorted(prompts_dir.iterdir()):
        if not slug_dir.is_dir():
            continue
        slug = slug_dir.name
        if specific_slugs and slug not in specific_slugs:
            continue
        if slug not in manifest and (slug_dir / "full.md").exists():
            needed.append(slug)
    return needed


# ──────────────────────────────────────────────────────────────────────────────
# Generation
# ──────────────────────────────────────────────────────────────────────────────

def generate_ambig(client, model: str, slug: str, prompts_dir: Path) -> str:
    """Generate ambig_metric.md for a single slug."""
    full_md = (prompts_dir / slug / "full.md").read_text()
    user_msg = REDACTION_USER.format(full_md=full_md)
    result = call_llm(client, REDACTION_SYSTEM, user_msg, model=model)
    return strip_code_fences(result)


def generate_manifest_entry(client, model: str, slug: str,
                            prompts_dir: Path, metrics_csv: dict) -> dict:
    """Generate a manifest entry for a single slug."""
    full_md = (prompts_dir / slug / "full.md").read_text()
    csv_row = metrics_csv.get(slug, {})
    user_msg = MANIFEST_USER.format(
        slug=slug,
        full_md=full_md[:8000],
        grader_name=csv_row.get("grader_name", "unknown"),
        eval_excerpt=csv_row.get("eval_excerpt", "N/A")[:2000],
    )
    raw = call_llm(client, MANIFEST_SYSTEM, user_msg, model=model)
    return json.loads(strip_code_fences(raw))


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark-dir", type=Path, required=True,
                    help="Benchmark directory (created by step_1_setup_benchmark.py in evaluate/ambig_ds_metric/)")
    ap.add_argument("--run", action="store_true",
                    help="Actually generate (default: dry-run)")
    ap.add_argument("--slugs", nargs="*", default=None,
                    help="Only process these slugs")
    ap.add_argument("--manifest-only", action="store_true",
                    help="Only update manifest, skip prompt generation")
    ap.add_argument("--prompts-only", action="store_true",
                    help="Only generate prompts, skip manifest")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"LLM model for generation (default: {DEFAULT_MODEL})")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing ambig_metric.md / manifest entries")
    args = ap.parse_args()

    benchmark_dir = args.benchmark_dir.resolve()
    prompts_dir = benchmark_dir / "prompts"
    if not prompts_dir.exists():
        sys.exit(f"prompts/ not found in {benchmark_dir}. Run evaluate/ambig_ds_metric/step_1_setup_benchmark.py first.")

    metrics_csv = load_metrics_csv(benchmark_dir)
    manifest = load_manifest(benchmark_dir)

    # Determine what needs doing
    if args.force and args.slugs:
        need_ambig = args.slugs if not args.manifest_only else []
        need_manifest = args.slugs if not args.prompts_only else []
    else:
        need_ambig = [] if args.manifest_only else find_slugs_needing_ambig(
            prompts_dir, args.slugs)
        need_manifest = [] if args.prompts_only else find_slugs_needing_manifest(
            manifest, prompts_dir, args.slugs)

    if not need_ambig and not need_manifest:
        print("Nothing to do — all prompts and manifest entries exist.")
        return

    print(f"=== Prompts to generate: {len(need_ambig)} ===")
    for s in need_ambig:
        print(f"  {s}")
    print(f"\n=== Manifest entries to generate: {len(need_manifest)} ===")
    for s in need_manifest:
        print(f"  {s}")

    if not args.run:
        print("\nDry run. Add --run to generate.")
        return

    client = make_client()

    # Generate ambig prompts
    for i, slug in enumerate(need_ambig, 1):
        print(f"\n[{i}/{len(need_ambig)}] Generating ambig_metric.md for {slug}...")
        try:
            result = generate_ambig(client, args.model, slug, prompts_dir)
            if not result.endswith("\n"):
                result += "\n"
            full_len = len((prompts_dir / slug / "full.md").read_text())
            ratio = len(result) / max(full_len, 1)
            if ratio < 0.5:
                # The redacted prompt should be roughly the same length as the
                # original (we only neutralize the metric section). A much
                # shorter output almost always means the LLM was truncated by
                # the gateway's max_tokens cap. Refuse to write so step 1 can
                # be re-run with a higher cap.
                print(f"  WARN: output is {ratio:.0%} of full.md length "
                      f"({len(result)} vs {full_len} chars) — likely truncated. "
                      f"Skipping write; re-run with a higher AMBIG_LLM_MAX_TOKENS.",
                      file=sys.stderr)
                continue
            out_path = prompts_dir / slug / "ambig_metric.md"
            out_path.write_text(result)
            print(f"  -> wrote {out_path} ({len(result)} chars)")
        except Exception as e:
            print(f"  ERROR: {e}", file=sys.stderr)

    # Generate manifest entries
    manifest_updated = False
    for i, slug in enumerate(need_manifest, 1):
        print(f"\n[{i}/{len(need_manifest)}] Generating manifest entry for {slug}...")
        try:
            entry = generate_manifest_entry(client, args.model, slug,
                                            prompts_dir, metrics_csv)
            manifest[slug] = entry
            manifest_updated = True
            print(f"  -> {entry['metric_name']} (lower_better={entry['is_lower_better']})")
        except Exception as e:
            print(f"  ERROR: {e}", file=sys.stderr)

    if manifest_updated:
        save_manifest(manifest, benchmark_dir)
        n_entries = sum(1 for k in manifest if not k.startswith("_"))
        print(f"\nManifest saved ({n_entries} entries)")

    print("\nDone. Review generated files before running experiments.")


if __name__ == "__main__":
    main()
