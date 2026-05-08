#!/usr/bin/env python3
"""Compile per-task audit verdicts into a single CSV + markdown report.

Reads <bench>/_verify/<slug>.json (one per slug) and emits:
  <bench>/_verify/audit_report.csv     one row per slug, with all 4 checks
  <bench>/_verify/audit_report.md      paper-ready table + per-task details

Usage:
    python compile_audit_report.py --benchmark-dir ./benchmark
"""
from __future__ import annotations
import argparse
import csv
import json
import sys
from pathlib import Path


def compile_report(benchmark_dir: Path) -> None:
    vdir = benchmark_dir / "_verify"
    manifest_path = benchmark_dir / "metric_manifest.json"
    if not manifest_path.exists():
        sys.exit(f"metric_manifest.json not found in {benchmark_dir}")
    man = json.loads(manifest_path.read_text())

    rows = []
    for jp in sorted(vdir.glob("*.json")):
        if jp.name.startswith("_"):
            continue
        d = json.loads(jp.read_text())
        slug = d["slug"]
        chks = d.get("checks", {})
        pa = chks.get("plausible_alternatives", {})
        ap = chks.get("ambiguity_preserved", {})
        dr = chks.get("decision_relevant", {})
        tp = chks.get("task_preserved", {})
        rows.append({
            "slug": slug,
            "verdict": d.get("verdict", "?"),
            "true_metric": man.get(slug, {}).get("metric_name", "?"),
            "task_preserved": "PASS" if tp.get("pass") else "FAIL",
            "task_preserved_why": tp.get("rationale", ""),
            "ambiguity_preserved": "PASS" if ap.get("pass") else "FAIL",
            "ambiguity_preserved_why": ap.get("rationale", ""),
            "leaked_cues": " | ".join(ap.get("leaked_cues", [])),
            "alternatives": " | ".join(pa.get("alternatives", [])),
            "alternatives_pass": "PASS" if pa.get("pass") else "FAIL",
            "alternatives_why": pa.get("rationale", ""),
            "decision_relevant": "PASS" if dr.get("pass") else "FAIL",
            "decision_relevant_why": dr.get("rationale", ""),
        })

    if not rows:
        print(f"No audit JSON files found in {vdir}")
        return

    # CSV
    csv_p = vdir / "audit_report.csv"
    with open(csv_p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Markdown
    n = len(rows)
    n_pass = sum(1 for r in rows if r["verdict"] == "pass")
    by_check = {k: sum(1 for r in rows if r[k] == "PASS")
                for k in ("task_preserved", "ambiguity_preserved",
                          "alternatives_pass", "decision_relevant")}

    md = []
    md.append(f"# Ambig-DS-Metric data quality audit\n")
    md.append(f"- Tasks audited: **{n}**\n")
    md.append(f"- Overall pass: **{n_pass}/{n}** ({100*n_pass/n:.0f}%)\n")
    md.append(f"- task_preserved:       {by_check['task_preserved']}/{n}")
    md.append(f"- ambiguity_preserved:  {by_check['ambiguity_preserved']}/{n}")
    md.append(f"- alternatives ≥ 2:     {by_check['alternatives_pass']}/{n}")
    md.append(f"- decision_relevant:    {by_check['decision_relevant']}/{n}\n")

    md.append("## Summary table\n")
    md.append("| slug | verdict | true metric | task | ambig | alts | dec.rel | alternatives |")
    md.append("|---|---|---|---|---|---|---|---|")
    for r in rows:
        md.append(f"| {r['slug']} | {r['verdict']} | {r['true_metric']} | "
                  f"{r['task_preserved']} | {r['ambiguity_preserved']} | "
                  f"{r['alternatives_pass']} | {r['decision_relevant']} | "
                  f"{r['alternatives']} |")

    md.append("\n## Failures\n")
    fails = [r for r in rows if r["verdict"] != "pass"]
    if not fails:
        md.append("_None — all tasks passed._\n")
    else:
        for r in fails:
            md.append(f"### {r['slug']}  (true metric: {r['true_metric']})")
            if r["task_preserved"] == "FAIL":
                md.append(f"- **task_preserved FAIL** — {r['task_preserved_why']}")
            if r["ambiguity_preserved"] == "FAIL":
                md.append(f"- **ambiguity_preserved FAIL** — {r['ambiguity_preserved_why']}")
                if r["leaked_cues"]:
                    md.append(f"  - leaked cues: {r['leaked_cues']}")
            if r["alternatives_pass"] == "FAIL":
                md.append(f"- **alternatives FAIL** — {r['alternatives_why']}")
            if r["decision_relevant"] == "FAIL":
                md.append(f"- **decision_relevant FAIL** — {r['decision_relevant_why']}")
            md.append("")

    (vdir / "audit_report.md").write_text("\n".join(md))
    print(f"Wrote {csv_p}")
    print(f"Wrote {vdir / 'audit_report.md'}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--benchmark-dir", type=Path, required=True,
                   help="Benchmark directory containing metric_manifest.json and _verify/")
    args = p.parse_args()
    compile_report(args.benchmark_dir.resolve())


if __name__ == "__main__":
    main()
