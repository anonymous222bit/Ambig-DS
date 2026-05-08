#!/usr/bin/env python3
"""LLM verification of target-ambig prompts against the paper checklist.

For every slug, an LLM judge reads the original DSBench prompt, the rewritten
target-ambig prompt produced by step_2_generate_ambig_prompts.py, and the
decoy manifest written by step_1_generate_decoy.py. It rates the ambiguous
variant on the four-item retention checklist from the paper
(Section 3.3, "Verification and Filtering"):

    1. Plausible alternatives
    2. Ambiguity preservation (no cue leaks across prompt + data package)
    3. Decision relevance
    4. Task preservation

For target ambiguity, the leak surface is wider than the metric suite: the
ambig prompt must (a) hide the original target-concept name and any unique
description of it, (b) not signal that two candidate targets exist, (c) not
re-introduce semantic feature names that were anonymised to ``f_01..f_NN``,
and (d) not state any numeric fact (row counts, feature counts) that
contradicts the manifest. The judge enforces these explicitly.

Inputs (per slug, all under ``$AMBIG_DSBENCH_ROOT``)::

    Dataset/data_modeling/data/data/task/<slug>.txt
    final_data_v3/target_ambig/data/<slug>/ambig_prompt.txt
    final_data_v3/target_ambig/data/<slug>/_manifest.json

Output (per slug)::

    final_data_v3/target_ambig/_verify/<slug>.json     structured judge verdict
    final_data_v3/target_ambig/_verify/_summary.json   aggregate pass/fail
    final_data_v3/target_ambig/_verify/rejected.txt    slugs that failed any check

The script never edits prompts, manifests, or the release; it only emits
judgements. step_4_build_release.py reads ``_verify/_summary.json`` if
present and (when invoked with ``--respect-verify``) skips slugs flagged
as failed.

Usage::

    # dry run (plan only)
    python step_2b_llm_verify.py --slugs bike-sharing-demand

    # actually call the LLM, single judge
    python step_2b_llm_verify.py --slugs bike-sharing-demand --run \
        --env-file /path/to/.env

    # paper protocol: 3-judge majority vote (App: cross_verifier_agreement)
    python step_2b_llm_verify.py --tasks_csv tasks.csv --run --n-judges 3

    # cross-verifier audit: rerun under a different family
    python step_2b_llm_verify.py --tasks_csv tasks.csv --run \
        --model gpt-5.4 --out-tag gpt

The default judge model is ``$AMBIG_VERIFIER_MODEL`` (falls back to
``$AMBIG_LLM_MODEL``, then to the package default).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd

from _llm_client import call_llm, make_client, DEFAULT_MODEL

DEFAULT_VERIFIER_MODEL = os.environ.get("AMBIG_VERIFIER_MODEL", DEFAULT_MODEL)


# --------------------------------------------------------------------------- #
# LLM prompt (target-ambiguity checklist)
# --------------------------------------------------------------------------- #
VERIFY_SYSTEM = """\
You are an expert reviewer for a benchmark on TARGET ambiguity in tabular
ML competitions. You receive (a) the original DSBench task prompt
(full.txt), (b) a rewritten variant (task_ambig.txt) that must hide the
original target column and add a second candidate target so that two
plausible targets coexist on disk, and (c) a manifest entry recording
which column is the true target and the anonymized->original feature
name map.

Apply the four-item retention checklist from the benchmark paper:

  1. PLAUSIBLE ALTERNATIVES — given only task_ambig.txt and the data
     package implied by it (anonymised features f_01..f_NN, plus two
     candidate target columns val_1 and val_2 in train.csv), at least
     two reasonable target columns remain consistent with the prompt.
     List them.

  2. AMBIGUITY PRESERVED — the ambiguous variant does NOT leak which
     column is the true target. Cue leaks include:
       * the original target concept name appearing in prose
         (e.g. "count", "price", "label", "target");
       * any synonym uniquely identifying the predicted concept
         (e.g. "rentals per hour", "house value");
       * explicit signposting that two candidate targets exist
         (the prompt must read as a single-target task);
       * mentions of "val_1", "val_2", "decoy", "ambiguous",
         "candidate target", "identify the target";
       * leftover semantic feature names from the manifest's
         feature_map (the prompt must use only f_01..f_NN);
       * sample_submission.csv being mentioned (it must be absent
         from the Files list);
       * numeric facts (n_train, n_test, n_features) that contradict
         the manifest — only the manifest values are allowed.

  3. DECISION RELEVANT — resolving the ambiguity changes a task-level
     choice a competent solver should make: which column is fit, which
     column appears in the submission file, and consequently which
     metric value the evaluator computes. (For target ambiguity this is
     true essentially by construction; rate fail only if the prompt
     somehow makes the choice trivial or moot.)

  4. TASK PRESERVED — the rewrite removes only target-identifying and
     feature-semantic information. Dataset narrative, evaluation metric
     description, submission-format shape (header order, row ordering,
     id column), and section headers are kept intact (modulo target
     anonymisation and `prediction` as the generic header column).

Output STRICT JSON, no markdown fences, no commentary. Schema::

  {
    "checks": {
      "plausible_alternatives": {
        "pass": true|false,
        "rationale": "<= 2 sentences",
        "alternatives": ["<col 1>", "<col 2>", ...]
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
ambiguity_preserved: any verbatim quote that names the original target
concept, names an original feature, signposts the two-target setup, or
contradicts the manifest counts is a leak.
"""

VERIFY_USER = """\
Slug: {slug}

Manifest (true target column and feature map are HIDDEN from the agent;
the judge gets them in order to detect leaks):
{manifest_block}

<full_txt>
{full_txt}
</full_txt>

<task_ambig_txt>
{ambig_txt}
</task_ambig_txt>
"""


# --------------------------------------------------------------------------- #
# Utils
# --------------------------------------------------------------------------- #
def strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines)
    return text


def manifest_block(m: dict) -> str:
    """Compact, judge-friendly view of the per-slug decoy manifest."""
    keys = (
        "task", "original_target_name", "target_type",
        "true_target_column", "decoy_column", "id_column",
        "n_train", "n_test", "n_features",
    )
    lines = [f"  {k}: {m.get(k, '')!r}" for k in keys]
    fmap = m.get("feature_map", {}) or {}
    if fmap:
        compact = ", ".join(f"{orig}->{anon}" for orig, anon in fmap.items())
        lines.append(f"  feature_map (must NOT appear in prompt): {compact}")
    diag = m.get("diagnostics", {}) or {}
    if "marginal_match_exact" in diag:
        lines.append(f"  diagnostics.marginal_match_exact: "
                     f"{diag['marginal_match_exact']!r}")
    if "correlation_truth_vs_decoy" in diag:
        lines.append(f"  diagnostics.|corr|: "
                     f"{abs(diag['correlation_truth_vs_decoy']):.3f}")
    return "\n".join(lines)


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


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #
def _verify_once(client, model: str, slug: str, full_txt: str,
                 ambig_txt: str, manifest: dict) -> dict:
    """Single judge call. Returns a validated verdict dict (always parseable)."""
    user = VERIFY_USER.format(
        slug=slug,
        manifest_block=manifest_block(manifest),
        full_txt=full_txt,
        ambig_txt=ambig_txt,
    )
    raw = call_llm(client, VERIFY_SYSTEM, user, model=model)
    if not raw:
        return {
            "error": "empty_llm_response",
            "verdict": "fail",
            "checks": {k: {"pass": False, "rationale": "no response"}
                       for k in ("plausible_alternatives", "ambiguity_preserved",
                                 "decision_relevant", "task_preserved")},
        }
    try:
        obj = json.loads(strip_code_fences(raw))
    except json.JSONDecodeError as e:
        return {
            "error": f"invalid_json: {e}",
            "raw": raw[:2000],
            "verdict": "fail",
            "checks": {k: {"pass": False, "rationale": "invalid json"}
                       for k in ("plausible_alternatives", "ambiguity_preserved",
                                 "decision_relevant", "task_preserved")},
        }
    return validate_verdict(obj)


def _aggregate_votes(votes: list[dict]) -> dict:
    """Per-check majority vote across N judge calls.

    For each of the 4 checks: pass iff strict majority voted pass. Even-count
    ties default to FAIL (conservative). Verdict = AND of checks.
    """
    needed = ("plausible_alternatives", "ambiguity_preserved",
              "decision_relevant", "task_preserved")
    n = len(votes)
    out_checks: dict = {}
    for k in needed:
        n_pass = sum(1 for v in votes
                     if v.get("checks", {}).get(k, {}).get("pass"))
        n_fail = n - n_pass
        majority_pass = n_pass > n_fail  # strict majority; ties -> FAIL
        rationales = [v["checks"].get(k, {}).get("rationale", "")
                      for v in votes
                      if v.get("checks", {}).get(k, {}).get("rationale")]
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


# --------------------------------------------------------------------------- #
# Path resolution
# --------------------------------------------------------------------------- #
def slug_paths(root: Path, slug: str) -> tuple[Path, Path, Path]:
    """Return (full_prompt, ambig_prompt, manifest) paths for a slug."""
    full_prompt = (root / "Dataset" / "data_modeling" / "data" / "data"
                   / "task" / f"{slug}.txt")
    ambig_prompt = (root / "final_data_v3" / "target_ambig" / "data" / slug
                    / "ambig_prompt.txt")
    manifest = (root / "final_data_v3" / "target_ambig" / "data" / slug
                / "_manifest.json")
    return full_prompt, ambig_prompt, manifest


def verify_slug(client, model: str, slug: str, root: Path,
                n_judges: int = 1) -> dict:
    full_p, ambig_p, manifest_p = slug_paths(root, slug)
    full_txt = full_p.read_text()
    ambig_txt = ambig_p.read_text()
    manifest = json.loads(manifest_p.read_text())
    n_judges = max(1, int(n_judges))
    votes = [_verify_once(client, model, slug, full_txt, ambig_txt, manifest)
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


def load_env_file(path: Path) -> None:
    """Minimal dotenv loader: same semantics as step_2's loader."""
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key.startswith(("OPENAI_", "AMBIG_")):
            os.environ[key] = val


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--slugs", nargs="+", help="One or more task slugs.")
    g.add_argument("--tasks_csv",
                   help="CSV with a 'task' column listing slugs.")
    ap.add_argument("--root", default=os.environ.get("AMBIG_DSBENCH_ROOT", ""),
                    help="Workspace root (default: $AMBIG_DSBENCH_ROOT).")
    ap.add_argument("--out-dir", default=None,
                    help="Override output dir. Default: "
                         "<root>/final_data_v3/target_ambig/_verify[_<tag>]/.")
    ap.add_argument("--out-tag", default=None,
                    help="Suffix for the default output dir; useful to keep "
                         "cross-verifier audits side by side "
                         "(e.g. --out-tag gpt).")
    ap.add_argument("--run", action="store_true",
                    help="Actually call the LLM (default: dry run).")
    ap.add_argument("--force", action="store_true",
                    help="Re-judge slugs that already have a verdict file.")
    ap.add_argument("--model", default=DEFAULT_VERIFIER_MODEL,
                    help=f"Judge model (default: {DEFAULT_VERIFIER_MODEL}).")
    ap.add_argument("--n-judges", type=int, default=1,
                    help="Independent judge calls per slug; per-check majority "
                         "vote with ties -> FAIL. Default: 1.")
    ap.add_argument("--env-file", default=None,
                    help="Optional .env file to load OPENAI_*/AMBIG_* vars.")
    args = ap.parse_args()

    if args.env_file:
        load_env_file(Path(args.env_file))
    if args.n_judges < 1:
        sys.exit("--n-judges must be >= 1")
    if not args.root:
        sys.exit("Set --root or AMBIG_DSBENCH_ROOT.")
    root = Path(args.root).resolve()

    if args.slugs:
        slugs = list(args.slugs)
    else:
        df = pd.read_csv(args.tasks_csv)
        if "task" not in df.columns:
            sys.exit(f"--tasks_csv must have a 'task' column "
                     f"(got {list(df.columns)})")
        slugs = df["task"].astype(str).tolist()

    if args.out_dir:
        out_dir = Path(args.out_dir).resolve()
    else:
        suffix = f"_verify_{args.out_tag}" if args.out_tag else "_verify"
        out_dir = (root / "final_data_v3" / "target_ambig" / suffix).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Plan: skip slugs missing inputs; skip already-judged slugs unless --force.
    todo: list[str] = []
    skipped_missing: list[str] = []
    for slug in slugs:
        full_p, ambig_p, manifest_p = slug_paths(root, slug)
        if not (full_p.exists() and ambig_p.exists() and manifest_p.exists()):
            skipped_missing.append(slug)
            continue
        if not args.force and (out_dir / f"{slug}.json").exists():
            continue
        todo.append(slug)

    print(f"=== Verify plan: {len(todo)} / {len(slugs)} tasks  "
          f"(model={args.model}, n_judges={args.n_judges}, out={out_dir}) ===")
    for s in todo:
        print(f"  {s}")
    if skipped_missing:
        print(f"  skipped {len(skipped_missing)} (missing prompt/manifest): "
              f"{skipped_missing[:5]}{' ...' if len(skipped_missing) > 5 else ''}")
    if not args.run:
        print("\nDry run. Add --run to execute.")
        return 0

    # Read base_url AFTER load_env_file so --env-file values win.
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    client = make_client(base_url=base_url)
    for i, slug in enumerate(todo, 1):
        print(f"\n[{i}/{len(todo)}] verifying {slug} ...")
        try:
            result = verify_slug(client, args.model, slug, root,
                                 n_judges=args.n_judges)
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}", file=sys.stderr)
            continue
        (out_dir / f"{slug}.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n"
        )
        flags = [k for k, v in result["checks"].items() if not v["pass"]]
        print(f"  -> {result['verdict'].upper()}"
              + (f"  failed: {flags}" if flags else ""))

    # Aggregate every verdict file present (not just this run's todo).
    needed = ("plausible_alternatives", "ambiguity_preserved",
              "decision_relevant", "task_preserved")
    summary: dict = {
        "model": args.model,
        "n_judges": args.n_judges,
        "n": 0, "pass": 0, "fail": 0,
        "by_check_fail": {k: 0 for k in needed},
        "verdicts": {},
    }
    rejected: list[str] = []
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
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    (out_dir / "rejected.txt").write_text(
        "\n".join(rejected) + ("\n" if rejected else "")
    )

    print(f"\n=== Verify summary ({args.model}) ===")
    print(f"  pass: {summary['pass']} / {summary['n']}")
    print(f"  fail: {summary['fail']}")
    for k, n in summary["by_check_fail"].items():
        if n:
            print(f"    {k}: {n}")
    if rejected:
        print(f"  rejected slugs written to {out_dir / 'rejected.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
