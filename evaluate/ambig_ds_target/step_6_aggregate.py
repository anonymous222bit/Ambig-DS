#!/usr/bin/env python3
"""Step 6 — aggregate per-task _grade.json files into paper-style results.

Reads <bench>/results/<run-name>/<slug>/_grade.json for the three runs that
correspond to a single (model, agent) pair:

    <run_prefix>_full
    <run_prefix>_ambig_target
    <run_prefix>_ambig_target_clarify

(where <run_prefix> typically looks like opencode_<model>) and computes the
paper's headline statistics on the RPG-normalized scores stored in
_grade.json["score_rpg"]:

    S_full, S_ambig, S_ask          (macro mean over paired tasks)
    delta_ambig = S_ambig - S_full
    delta_ask   = S_ask   - S_ambig
    one-sided paired Wilcoxon       (Full > Ambig.;  Ask > Ambig.)
    paired bootstrap 95% CIs        (delta_ambig, delta_ask)

Only tasks present in ALL THREE runs (and with a non-null score_rpg in each)
are kept; the count is reported.

Usage:
    python step_6_aggregate.py --benchmark-dir ./benchmark \\
        --run-prefix opencode_gemini_3_flash

    # Multiple models in one call:
    python step_6_aggregate.py --benchmark-dir ./benchmark \\
        --run-prefix opencode_gemini_3_flash,opencode_anthropic_claude_haiku_4_5_v1_0

    # Custom suffixes (defaults below mirror step_4 / step_5):
    python step_6_aggregate.py --benchmark-dir ./benchmark \\
        --run-prefix opencode_gemini_3_flash \\
        --suffix-full _full --suffix-ambig _ambig_target \\
        --suffix-ask _ambig_target_clarify

Output: prints a table to stdout and writes
    <bench>/results/_aggregate/<run_prefix>.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

import numpy as np


# --------------------------------------------------------------------------- #
def _load_run(results_dir: Path, run_name: str) -> dict[str, float]:
    """Map slug -> score_rpg for one run directory. Skips missing/null."""
    out: dict[str, float] = {}
    rdir = results_dir / run_name
    if not rdir.exists():
        return out
    for grade_p in sorted(rdir.glob("*/_grade.json")):
        slug = grade_p.parent.name
        try:
            g = json.loads(grade_p.read_text())
        except Exception:
            continue
        rpg = g.get("score_rpg")
        if rpg is None:
            continue
        try:
            out[slug] = float(rpg)
        except (TypeError, ValueError):
            continue
    return out


# --------------------------------------------------------------------------- #
def _wilcoxon_one_sided_greater(a: list[float], b: list[float]) -> float | None:
    """One-sided paired Wilcoxon signed-rank H1: median(a - b) > 0.

    Returns p-value, or None if scipy is unavailable or the sample is degenerate.
    """
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        return None
    diffs = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if len(diffs) == 0 or np.all(diffs == 0):
        return None
    try:
        res = wilcoxon(diffs, alternative="greater", zero_method="wilcox")
        return float(res.pvalue)
    except Exception:
        return None


def _paired_bootstrap_ci(a: list[float], b: list[float], n_boot: int = 10_000,
                         alpha: float = 0.05, seed: int = 0
                         ) -> tuple[float, float] | None:
    """Bootstrap 95% CI for mean(a - b), resampling task indices with replacement."""
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    if len(a_arr) == 0:
        return None
    rng = np.random.default_rng(seed)
    n = len(a_arr)
    deltas = a_arr - b_arr
    idx = rng.integers(0, n, size=(n_boot, n))
    samples = deltas[idx].mean(axis=1)
    lo = float(np.quantile(samples, alpha / 2))
    hi = float(np.quantile(samples, 1 - alpha / 2))
    return lo, hi


# --------------------------------------------------------------------------- #
def aggregate_one(benchmark_dir: Path, run_prefix: str,
                  suffix_full: str, suffix_ambig: str, suffix_ask: str,
                  n_boot: int) -> dict:
    results = benchmark_dir / "results"
    full = _load_run(results, run_prefix + suffix_full)
    ambig = _load_run(results, run_prefix + suffix_ambig)
    ask = _load_run(results, run_prefix + suffix_ask)

    paired = sorted(set(full) & set(ambig) & set(ask))
    s_full = [full[s] for s in paired]
    s_ambig = [ambig[s] for s in paired]
    s_ask = [ask[s] for s in paired]

    rep: dict = {
        "run_prefix": run_prefix,
        "n_paired": len(paired),
        "n_full": len(full),
        "n_ambig": len(ambig),
        "n_ask": len(ask),
        "tasks": paired,
    }
    if not paired:
        rep["error"] = (
            f"no paired tasks across {run_prefix}{{{suffix_full},"
            f"{suffix_ambig},{suffix_ask}}} — check run names / score_rpg."
        )
        return rep

    rep["S_full"] = mean(s_full)
    rep["S_ambig"] = mean(s_ambig)
    rep["S_ask"] = mean(s_ask)
    rep["delta_ambig"] = rep["S_ambig"] - rep["S_full"]
    rep["delta_ask"] = rep["S_ask"] - rep["S_ambig"]

    p_fa = _wilcoxon_one_sided_greater(s_full, s_ambig)   # H1: full > ambig
    p_ka = _wilcoxon_one_sided_greater(s_ask, s_ambig)    # H1: ask  > ambig
    rep["wilcoxon_full_gt_ambig_p"] = p_fa
    rep["wilcoxon_ask_gt_ambig_p"] = p_ka

    ci_da = _paired_bootstrap_ci(s_ambig, s_full, n_boot=n_boot, seed=0)
    ci_dk = _paired_bootstrap_ci(s_ask, s_ambig, n_boot=n_boot, seed=1)
    if ci_da:
        rep["delta_ambig_ci95"] = list(ci_da)
    if ci_dk:
        rep["delta_ask_ci95"] = list(ci_dk)
    return rep


# --------------------------------------------------------------------------- #
def _fmt(x, fmt="{:+.3f}"):
    return "—" if x is None else fmt.format(x)


def _print_table(reports: list[dict]) -> None:
    hdr = (
        f"{'Run':38s} {'n':>4s} "
        f"{'S_full':>7s} {'S_ambig':>8s} {'S_ask':>7s} "
        f"{'Δ_ambig':>9s} {'Δ_ask':>8s} "
        f"{'p(F>A)':>10s} {'p(K>A)':>10s} "
        f"{'CI Δ_ambig':>20s} {'CI Δ_ask':>20s}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in reports:
        if "error" in r:
            print(f"{r['run_prefix']:38s} ERROR: {r['error']}")
            continue
        ci_a = (r.get("delta_ambig_ci95") or [None, None])
        ci_k = (r.get("delta_ask_ci95") or [None, None])
        ci_a_s = "—" if ci_a[0] is None else f"[{ci_a[0]:+.3f}, {ci_a[1]:+.3f}]"
        ci_k_s = "—" if ci_k[0] is None else f"[{ci_k[0]:+.3f}, {ci_k[1]:+.3f}]"
        print(
            f"{r['run_prefix']:38s} {r['n_paired']:>4d} "
            f"{r['S_full']:>7.3f} {r['S_ambig']:>8.3f} {r['S_ask']:>7.3f} "
            f"{r['delta_ambig']:>+9.3f} {r['delta_ask']:>+8.3f} "
            f"{_fmt(r['wilcoxon_full_gt_ambig_p'], '{:>10.4f}'):>10s} "
            f"{_fmt(r['wilcoxon_ask_gt_ambig_p'],  '{:>10.4f}'):>10s} "
            f"{ci_a_s:>20s} {ci_k_s:>20s}"
        )


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark-dir", type=Path, required=True)
    ap.add_argument("--run-prefix", required=True,
                    help="Comma-separated list of <agent>_<model> prefixes "
                         "(e.g. opencode_gemini_3_flash). The script appends "
                         "the variant suffixes to find each run dir.")
    ap.add_argument("--suffix-full", default="_full")
    ap.add_argument("--suffix-ambig", default="_ambig_target")
    ap.add_argument("--suffix-ask", default="_ambig_target_clarify")
    ap.add_argument("--n-boot", type=int, default=10_000)
    args = ap.parse_args()

    bench = args.benchmark_dir.resolve()
    out_dir = bench / "results" / "_aggregate"
    out_dir.mkdir(parents=True, exist_ok=True)

    prefixes = [p.strip() for p in args.run_prefix.split(",") if p.strip()]
    reports = []
    for prefix in prefixes:
        rep = aggregate_one(bench, prefix, args.suffix_full, args.suffix_ambig,
                            args.suffix_ask, args.n_boot)
        reports.append(rep)
        out_p = out_dir / f"{prefix}.json"
        out_p.write_text(json.dumps(rep, indent=2))

    _print_table(reports)
    print(f"\nWrote per-prefix JSON to {out_dir}")


if __name__ == "__main__":
    main()
