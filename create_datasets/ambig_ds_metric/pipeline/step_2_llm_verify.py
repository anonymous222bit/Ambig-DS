#!/usr/bin/env python3
"""LLM verification of ambig_metric.md prompts against the paper checklist.

For every slug, an LLM judge reads `full.md`, `ambig_metric.md`, and the
manifest entry and rates the ambiguous variant on the four-item checklist
from the paper (Section 3.3, "Verification and Filtering"):

    1. Plausible alternatives
    2. Ambiguity preservation (no cue leaks across prompt + submission format)
    3. Decision relevance
    4. Task preservation

Output (per slug):
    <bench>/_verify/<slug>.json      structured judge verdict
    <bench>/_verify/_summary.json    aggregate pass/fail counts + per-slug verdicts
    <bench>/_verify/rejected.txt     slugs that failed any check (one per line)

The script never edits prompts or the manifest; it only emits judgements.
Step 3 (upload) reads `_verify/_summary.json` if present and refuses to
upload slugs flagged as failed unless `--allow-failed` is given.

Usage:
    python step_2_llm_verify.py --benchmark-dir ../benchmark                   # dry run (plan)
    python step_2_llm_verify.py --benchmark-dir ../benchmark --run             # all missing
    python step_2_llm_verify.py --benchmark-dir ../benchmark --run --slugs A B # subset
    python step_2_llm_verify.py --benchmark-dir ../benchmark --run --force     # re-judge all

The default judge model is `AMBIG_VERIFIER_MODEL` (falls back to
`AMBIG_LLM_MODEL`, then to `gpt-4o-mini`). To match the paper's
cross-verifier audit (non-Claude families), pass `--model gpt-4o` and/or
`--model gemini-2.5-pro` and write the verdicts under
`<bench>/_verify_<tag>/` via `--out-tag <tag>`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from _llm_client import call_llm, make_client, DEFAULT_MODEL

DEFAULT_VERIFIER_MODEL = os.environ.get("AMBIG_VERIFIER_MODEL", DEFAULT_MODEL)

# ──────────────────────────────────────────────────────────────────────────────
# LLM prompt
# ──────────────────────────────────────────────────────────────────────────────

VERIFY_SYSTEM = """\
You are an expert reviewer for a benchmark on metric ambiguity in ML
competitions. You receive (a) the original Kaggle competition
description (full.md), (b) a redacted version (ambig_metric.md) that
must hide the evaluation metric, and (c) the structured manifest entry
recording the true metric.

Apply the four-item retention checklist from the benchmark paper:

  1. PLAUSIBLE ALTERNATIVES — given only ambig_metric.md and the data
     package implied by it (column names, sample submission, etc.), at
     least two reasonable evaluation metrics remain consistent with the
     prompt. List them.

  2. AMBIGUITY PRESERVED — the ambiguous variant does NOT leak the true
     metric anywhere: not in the Evaluation section, not in inline
     mentions ("predict a probability", "minimize ...", explicit metric
     names), and not in the submission-format hints. Cue leaks include
     formula fragments, optimization-direction wording, probability/
     hard-label hints that uniquely identify the metric, and
     metric-specific column semantics. Submission column NAMES are
     allowed to remain (the grader needs them); their METRIC-IDENTIFYING
     SEMANTICS are not.

  3. DECISION RELEVANT — resolving the ambiguity changes a task-level
     choice a competent solver should make: hard labels vs probabilities,
     optimization direction, threshold/ranking behavior, top-K, clipping,
     column-wise aggregation, or submission semantics.

  4. TASK PRESERVED — the redaction removes only metric-related
     information. Data descriptions, file lists, column definitions,
     submission column names, timeline, prizes, and citation are kept
     intact (modulo neutralized metric phrasing).

Output STRICT JSON, no markdown fences, no commentary. Schema:

{
  "checks": {
    "plausible_alternatives": {
      "pass": true|false,
      "rationale": "<= 2 sentences",
      "alternatives": ["<metric 1>", "<metric 2>", ...]
    },
    "ambiguity_preserved": {
      "pass": true|false,
      "rationale": "<= 2 sentences",
      "leaked_cues": ["<verbatim quote 1>", ...]
    },
    "decision_relevant": {
      "pass": true|false,
      "rationale": "<= 2 sentences"
    },
    "task_preserved": {
      "pass": true|false,
      "rationale": "<= 2 sentences"
    }
  },
  "verdict": "pass" | "fail",
  "notes": "optional <= 2 sentences"
}

`verdict` is "pass" iff all four checks pass. Be strict on
ambiguity_preserved: any verbatim quote that names or formulaically
identifies the true metric is a leak.
"""

VERIFY_USER = """\
Slug: {slug}

True metric (manifest):
{manifest_block}

<full_md>
{full_md}
</full_md>

<ambig_metric_md>
{ambig_md}
</ambig_metric_md>
"""


# ──────────────────────────────────────────────────────────────────────────────
# Utils
# ──────────────────────────────────────────────────────────────────────────────

def strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines)
    return text


def load_manifest(benchmark_dir: Path) -> dict:
    for p in (benchmark_dir / "prompts" / "_metric_manifest.json",
              benchmark_dir / "metric_manifest.json"):
        if p.exists():
            return json.loads(p.read_text())
    return {}


def manifest_block(entry: dict) -> str:
    keys = ("metric_name", "metric_description", "submission_format",
            "is_lower_better", "notes")
    return "\n".join(f"  {k}: {entry.get(k, '')!r}" for k in keys)


def validate_verdict(obj: dict) -> dict:
    """Best-effort coercion of judge output into the canonical schema."""
    checks = obj.get("checks", {}) or {}
    needed = ("plausible_alternatives", "ambiguity_preserved",
              "decision_relevant", "task_preserved")
    out_checks = {}
    for k in needed:
        c = checks.get(k, {}) or {}
        out_checks[k] = {
            "pass": bool(c.get("pass", False)),
            "rationale": str(c.get("rationale", "")),
        }
        if k == "plausible_alternatives":
            alts = c.get("alternatives", []) or []
            out_checks[k]["alternatives"] = [str(a) for a in alts]
        if k == "ambiguity_preserved":
            cues = c.get("leaked_cues", []) or []
            out_checks[k]["leaked_cues"] = [str(x) for x in cues]
    derived = all(out_checks[k]["pass"] for k in needed)
    verdict = obj.get("verdict")
    if verdict not in {"pass", "fail"}:
        verdict = "pass" if derived else "fail"
    return {
        "checks": out_checks,
        "verdict": verdict,
        "notes": str(obj.get("notes", "")),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Verification
# ──────────────────────────────────────────────────────────────────────────────

def _verify_once(client, model: str, slug: str, full_md: str, ambig_md: str,
                 entry: dict) -> dict:
    """Single judge call. Returns a validated verdict dict (always parseable)."""
    user = VERIFY_USER.format(
        slug=slug,
        manifest_block=manifest_block(entry),
        full_md=full_md,
        ambig_md=ambig_md,
    )
    raw = call_llm(client, VERIFY_SYSTEM, user, model=model)
    if not raw:
        return {"error": "empty_llm_response", "verdict": "fail",
                "checks": {k: {"pass": False, "rationale": "no response"}
                           for k in ("plausible_alternatives", "ambiguity_preserved",
                                     "decision_relevant", "task_preserved")}}
    try:
        obj = json.loads(strip_code_fences(raw))
    except json.JSONDecodeError as e:
        return {"error": f"invalid_json: {e}", "raw": raw[:2000], "verdict": "fail",
                "checks": {k: {"pass": False, "rationale": "invalid json"}
                           for k in ("plausible_alternatives", "ambiguity_preserved",
                                     "decision_relevant", "task_preserved")}}
    return validate_verdict(obj)


def _aggregate_votes(votes: list[dict]) -> dict:
    """Per-check majority vote across N judge calls.

    For each of the 4 checks: pass if strict majority of judges voted pass.
    Even-count ties default to FAIL (conservative). Verdict = AND of checks.
    Aggregates rationales / alternatives / leaked_cues from all votes.
    """
    needed = ("plausible_alternatives", "ambiguity_preserved",
              "decision_relevant", "task_preserved")
    n = len(votes)
    out_checks = {}
    for k in needed:
        n_pass = sum(1 for v in votes if v.get("checks", {}).get(k, {}).get("pass"))
        n_fail = n - n_pass
        majority_pass = n_pass > n_fail  # strict majority; ties -> FAIL
        rationales = [v["checks"].get(k, {}).get("rationale", "")
                      for v in votes if v.get("checks", {}).get(k, {}).get("rationale")]
        entry = {
            "pass": majority_pass,
            "rationale": " || ".join(rationales)[:1500],
            "votes": {"pass": n_pass, "fail": n_fail, "n": n},
        }
        if k == "plausible_alternatives":
            alts: list[str] = []
            seen: set[str] = set()
            for v in votes:
                for a in v.get("checks", {}).get(k, {}).get("alternatives", []) or []:
                    key = a.strip().lower()
                    if key and key not in seen:
                        seen.add(key)
                        alts.append(a)
            entry["alternatives"] = alts
        if k == "ambiguity_preserved":
            cues: list[str] = []
            seen = set()
            for v in votes:
                for c in v.get("checks", {}).get(k, {}).get("leaked_cues", []) or []:
                    if c not in seen:
                        seen.add(c)
                        cues.append(c)
            entry["leaked_cues"] = cues
        out_checks[k] = entry
    verdict = "pass" if all(out_checks[k]["pass"] for k in needed) else "fail"
    notes = " || ".join(v.get("notes", "") for v in votes if v.get("notes"))
    return {"checks": out_checks, "verdict": verdict, "notes": notes[:1500]}


def verify_slug(client, model: str, slug: str, prompts_dir: Path,
                manifest: dict, n_judges: int = 1) -> dict:
    full_md = (prompts_dir / slug / "full.md").read_text()
    ambig_md = (prompts_dir / slug / "ambig_metric.md").read_text()
    entry = manifest.get(slug, {})
    n_judges = max(1, int(n_judges))
    votes = [_verify_once(client, model, slug, full_md, ambig_md, entry)
             for _ in range(n_judges)]
    if n_judges == 1:
        out = votes[0]
    else:
        out = _aggregate_votes(votes)
        out["votes_raw"] = votes
    out["slug"] = slug
    out["model"] = model
    out["n_judges"] = n_judges
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark-dir", type=Path, required=True)
    ap.add_argument("--run", action="store_true",
                    help="Actually call the LLM (default: dry run)")
    ap.add_argument("--slugs", nargs="*", default=None,
                    help="Only verify these slugs")
    ap.add_argument("--force", action="store_true",
                    help="Re-judge slugs that already have a verdict")
    ap.add_argument("--model", default=DEFAULT_VERIFIER_MODEL,
                    help=f"Judge model (default: {DEFAULT_VERIFIER_MODEL})")
    ap.add_argument("--out-tag", default=None,
                    help="Write under <bench>/_verify_<tag>/ instead of "
                         "<bench>/_verify/. Use this to keep cross-verifier "
                         "audits (gpt, gemini, ...) side by side.")
    ap.add_argument("--n-judges", type=int, default=1,
                    help="Number of independent judge calls per slug. "
                         "Each of the 4 checks is decided by strict majority "
                         "vote across the N votes (ties default to FAIL). "
                         "Default: 1.")
    args = ap.parse_args()
    if args.n_judges < 1:
        sys.exit("--n-judges must be >= 1")

    benchmark_dir = args.benchmark_dir.resolve()
    prompts_dir = benchmark_dir / "prompts"
    if not prompts_dir.exists():
        sys.exit(f"prompts/ not found in {benchmark_dir}")

    task_list = (benchmark_dir / "task_list.txt").read_text().strip().splitlines()
    if args.slugs:
        unknown = sorted(set(args.slugs) - set(task_list))
        if unknown:
            sys.exit(f"slugs not in task_list.txt: {unknown}")
        task_list = [s for s in task_list if s in args.slugs]

    manifest = load_manifest(benchmark_dir)
    out_dir = benchmark_dir / (f"_verify_{args.out_tag}" if args.out_tag else "_verify")
    out_dir.mkdir(parents=True, exist_ok=True)

    todo = []
    for slug in task_list:
        if not (prompts_dir / slug / "ambig_metric.md").exists():
            print(f"  skip {slug}: ambig_metric.md missing")
            continue
        if not args.force and (out_dir / f"{slug}.json").exists():
            continue
        todo.append(slug)

    print(f"=== Verify plan: {len(todo)} / {len(task_list)} tasks "
          f"(model={args.model}, n_judges={args.n_judges}, out={out_dir}) ===")
    for s in todo:
        print(f"  {s}")
    if not args.run:
        print("\nDry run. Add --run to execute.")
        return

    client = make_client()
    for i, slug in enumerate(todo, 1):
        print(f"\n[{i}/{len(todo)}] verifying {slug} ...")
        try:
            result = verify_slug(client, args.model, slug, prompts_dir, manifest,
                                 n_judges=args.n_judges)
        except Exception as e:
            print(f"  ERROR: {e}", file=sys.stderr)
            continue
        (out_dir / f"{slug}.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        flags = [k for k, v in result["checks"].items() if not v["pass"]]
        print(f"  -> {result['verdict'].upper()}"
              + (f"  failed: {flags}" if flags else ""))

    # Aggregate
    summary = {"model": args.model, "n_judges": args.n_judges,
               "n": 0, "pass": 0, "fail": 0,
               "by_check_fail": {k: 0 for k in (
                   "plausible_alternatives", "ambiguity_preserved",
                   "decision_relevant", "task_preserved")},
               "verdicts": {}}
    rejected = []
    for path in sorted(out_dir.glob("*.json")):
        if path.name.startswith("_"):
            continue
        v = json.loads(path.read_text())
        summary["n"] += 1
        summary["verdicts"][v["slug"]] = v["verdict"]
        if v["verdict"] == "pass":
            summary["pass"] += 1
        else:
            summary["fail"] += 1
            rejected.append(v["slug"])
            for k, c in v.get("checks", {}).items():
                if not c.get("pass", False) and k in summary["by_check_fail"]:
                    summary["by_check_fail"][k] += 1
    (out_dir / "_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    (out_dir / "rejected.txt").write_text("\n".join(rejected) + ("\n" if rejected else ""))

    print(f"\n=== Verify summary ({args.model}) ===")
    print(f"  pass: {summary['pass']} / {summary['n']}")
    print(f"  fail: {summary['fail']}")
    for k, n in summary["by_check_fail"].items():
        if n:
            print(f"    {k}: {n}")
    if rejected:
        print(f"  rejected slugs written to {out_dir / 'rejected.txt'}")


if __name__ == "__main__":
    main()
