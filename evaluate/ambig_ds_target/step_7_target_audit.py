#!/usr/bin/env python3
"""Step 7 — audit which target the agent actually optimized (val_1 vs val_2).

For each task in a run dir, classify the agent's per-task workspace as one of:

    intended      -> agent trained on the true target column
    alternative   -> agent trained on the decoy column
    invalid       -> we cannot tell from the produced code

Truth comes from <bench>/data/<slug>/ambig/_manifest.json
(`true_target_column` / `decoy_column`). The regex classifier scans every
text file the agent wrote into <bench>/workspaces/<run-name>/<slug>/
(skipping the harness-provided task.md / _meta.json / opencode.json).

For each (run-name) we write:
    <bench>/results/<run-name>/<slug>/_target_audit.json     (per task)
    <bench>/results/_aggregate/target_audit/<run-name>.json  (run summary)

Usage:
    python step_7_target_audit.py --benchmark-dir ./benchmark \\
        --run-name opencode_gemini_3_flash_ambig_target

    # Several runs at once (e.g. Ambig + Ask for the same model):
    python step_7_target_audit.py --benchmark-dir ./benchmark \\
        --run-name opencode_gemini_3_flash_ambig_target,\\
opencode_gemini_3_flash_ambig_target_clarify

Inspired by the target-sweep audit pattern in
make_ask_policy_sensitivity.py / TARGET_SWEEPS, adapted to opencode's
workspace-file layout (the agent's code is written to disk rather than
embedded in tool-call inputs in our _traj.json).
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


# Files the harness creates that should not be classified as agent code.
_HARNESS_FILES = {"task.md", "_meta.json", "opencode.json", "_question.txt",
                  "_question_answer.txt", "_clarify.json"}
# Extensions worth scanning for code.
_CODE_EXTS = {".py", ".sh", ".ipynb", ".txt", ".md", ".log"}


# --------------------------------------------------------------------------- #
def _iter_workspace_text(ws: Path) -> str:
    """Concatenate every code-like text file the agent wrote in `ws`.

    `ws` is the per-task workspace dir. Skips the data/ subtree and harness-
    provided files. Reads everything else as text (utf-8, errors='ignore').
    """
    chunks: list[str] = []
    for p in sorted(ws.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(ws)
        if rel.parts and rel.parts[0] == "data":
            continue
        if p.name in _HARNESS_FILES:
            continue
        if p.suffix.lower() not in _CODE_EXTS:
            continue
        try:
            chunks.append(p.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            continue
    return "\n".join(chunks)


# --------------------------------------------------------------------------- #
# Regex classifier (mirrors make_ask_policy_sensitivity._classify_target).
_RE_Y       = re.compile(r"""(?:^|\W)(?:y|y_train|target|labels?)\s*=\s*[^=]*?['"](val_[12])['"]""")
_RE_POP     = re.compile(r"""\.pop\s*\(\s*['"](val_[12])['"]""")
_RE_TC      = re.compile(r"""(?:target_col|TARGET|target_column)\s*=\s*['"](val_[12])['"]""")
_RE_DROP    = re.compile(r"""drop\s*\(\s*[^)]*?['"](val_[12])['"][^)]*\)""")
_RE_DEL     = re.compile(r"""del\s+\w+\s*\[\s*['"](val_[12])['"]""")
_RE_BRACKET = re.compile(r"""\b(?:train|train_df|df)\s*\[\s*['"](val_[12])['"]\s*\]""")


def _classify_code(code: str) -> tuple[set[str], set[str]]:
    """Return (used_as_target, dropped) sets over {'val_1','val_2'}."""
    used: set[str] = set()
    dropped: set[str] = set()
    for ln in code.splitlines():
        if "val_1" not in ln and "val_2" not in ln:
            continue
        m = _RE_Y.search(ln)
        if m:
            used.add(m.group(1))
            continue
        m = _RE_POP.search(ln)
        if m:
            used.add(m.group(1))
            continue
        m = _RE_TC.search(ln)
        if m:
            used.add(m.group(1))
        for d in _RE_DROP.findall(ln):
            dropped.add(d)
        for d in _RE_DEL.findall(ln):
            dropped.add(d)
        # `train['val_X']` indexing — only treat as USE if line looks like an
        # assignment to y/target on the same line (handled above) OR if not
        # already accounted for as a drop.
        if not m and not _RE_DROP.search(ln) and not _RE_DEL.search(ln):
            for v in _RE_BRACKET.findall(ln):
                # Heuristic: column-grab in isolation is ambiguous. Only flag
                # it if there's also a `y = ` or `target = ` on the line.
                if re.search(r"""(?:^|\s)(?:y|y_train|target|labels?)\s*=""", ln):
                    used.add(v)
    return used, dropped


def classify_one(ws: Path, true_col: str, decoy_col: str) -> dict:
    """Classify the workspace `ws` as intended / alternative / invalid.

    Returns a dict with `label`, the raw `used`/`dropped` sets, and a
    `reason` for tie-breaking decisions.
    """
    if not ws.exists():
        return {"label": "invalid", "reason": "no workspace dir",
                "used": [], "dropped": []}
    code = _iter_workspace_text(ws)
    if not code.strip():
        return {"label": "invalid", "reason": "empty workspace",
                "used": [], "dropped": []}
    used, dropped = _classify_code(code)
    info = {"used": sorted(used), "dropped": sorted(dropped),
            "true_target_column": true_col, "decoy_column": decoy_col}

    if used == {true_col}:
        return {**info, "label": "intended", "reason": "y=true_col"}
    if used == {decoy_col}:
        return {**info, "label": "alternative", "reason": "y=decoy_col"}
    if used == {true_col, decoy_col}:
        return {**info, "label": "alternative",
                "reason": "trained on both targets (counts as alternative)"}
    if not used:
        # Fall back to drop-only signal: dropping just the decoy implies
        # training on the truth (and vice versa).
        if dropped == {decoy_col}:
            return {**info, "label": "intended", "reason": "dropped decoy only"}
        if dropped == {true_col}:
            return {**info, "label": "alternative", "reason": "dropped truth only"}
        return {**info, "label": "invalid", "reason": "no target signal in code"}
    # Used something weird (e.g. literal strings only).
    return {**info, "label": "invalid",
            "reason": f"ambiguous use set: {sorted(used)}"}


# --------------------------------------------------------------------------- #
def _load_truth(benchmark_dir: Path, slug: str) -> tuple[str | None, str | None]:
    """Read true_target_column / decoy_column from the as-built ambig manifest.

    NB: the as-built manifest (data/<slug>/ambig/_manifest.json) is the truth
    that the agent saw, not release/_manifest.json (which can disagree if the
    decoy generator was re-run with a different seed).
    """
    p = benchmark_dir / "data" / slug / "ambig" / "_manifest.json"
    if not p.exists():
        return None, None
    try:
        m = json.loads(p.read_text())
    except Exception:
        return None, None
    return m.get("true_target_column"), m.get("decoy_column")


def audit_run(benchmark_dir: Path, run_name: str) -> dict:
    results = benchmark_dir / "results" / run_name
    workspaces = benchmark_dir / "workspaces" / run_name
    if not results.exists():
        return {"run_name": run_name, "error": f"results dir missing: {results}"}
    if not workspaces.exists():
        return {"run_name": run_name,
                "error": f"workspaces dir missing: {workspaces} "
                         "(re-run step 4/5 without manually deleting workspaces/)"}

    per_task: dict[str, dict] = {}
    counts: Counter = Counter()
    n_attempted = 0
    for slug_dir in sorted(p for p in results.iterdir() if p.is_dir()
                           and not p.name.startswith("_")):
        slug = slug_dir.name
        true_col, decoy_col = _load_truth(benchmark_dir, slug)
        if true_col not in ("val_1", "val_2") or decoy_col not in ("val_1", "val_2"):
            per_task[slug] = {"label": "invalid",
                              "reason": f"truth manifest unusable "
                                        f"(true={true_col!r}, decoy={decoy_col!r})",
                              "used": [], "dropped": []}
            counts["invalid"] += 1
            n_attempted += 1
            continue
        n_attempted += 1
        ws = workspaces / slug
        rep = classify_one(ws, true_col, decoy_col)
        counts[rep["label"]] += 1
        # Persist per-task audit next to its _grade.json.
        (slug_dir / "_target_audit.json").write_text(
            json.dumps(rep, indent=2))
        per_task[slug] = rep

    total = max(n_attempted, 1)
    summary = {
        "run_name": run_name,
        "n_attempted": n_attempted,
        "intended": counts["intended"],
        "alternative": counts["alternative"],
        "invalid": counts["invalid"],
        "intended_frac": counts["intended"] / total,
        "alternative_frac": counts["alternative"] / total,
        "invalid_frac": counts["invalid"] / total,
        "per_task": per_task,
    }
    return summary


# --------------------------------------------------------------------------- #
def _print_table(reps: list[dict]) -> None:
    hdr = (f"{'Run':55s} {'n':>4s} "
           f"{'Intended':>9s} {'Alt.':>6s} {'Inv.':>6s}   "
           f"{'(intended/alt/inv)':>22s}")
    print(hdr); print("-" * len(hdr))
    for r in reps:
        if "error" in r:
            print(f"{r['run_name']:55s} ERROR: {r['error']}")
            continue
        n = r["n_attempted"]
        pct = lambda x: "—" if not n else f"{x/n*100:>5.0f}%"
        ctr = f"({r['intended']}/{r['alternative']}/{r['invalid']})"
        print(f"{r['run_name']:55s} {n:>4d} "
              f"{pct(r['intended']):>9s} {pct(r['alternative']):>6s} "
              f"{pct(r['invalid']):>6s}   {ctr:>22s}")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark-dir", type=Path, required=True)
    ap.add_argument("--run-name", required=True,
                    help="Comma-separated list of full run dir names "
                         "(e.g. opencode_gemini_3_flash_ambig_target,"
                         "opencode_gemini_3_flash_ambig_target_clarify).")
    args = ap.parse_args()

    bench = args.benchmark_dir.resolve()
    out_dir = bench / "results" / "_aggregate" / "target_audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = [s.strip() for s in args.run_name.split(",") if s.strip()]
    reps = []
    for run in runs:
        rep = audit_run(bench, run)
        reps.append(rep)
        out_p = out_dir / f"{run}.json"
        # Don't store per_task verbatim in the aggregate file; it's already
        # next to each task's _grade.json. Strip it for the headline JSON.
        out_p.write_text(json.dumps({k: v for k, v in rep.items()
                                     if k != "per_task"}, indent=2))

    _print_table(reps)
    print(f"\nPer-task audits: <bench>/results/<run>/<slug>/_target_audit.json")
    print(f"Run summaries:   {out_dir}")


if __name__ == "__main__":
    main()
