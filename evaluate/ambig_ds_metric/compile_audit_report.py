#!/usr/bin/env python3
"""Compile per-task audit verdicts into a single CSV + markdown report.

Reads <bench>/_verify/<slug>.json (one per slug) and emits:
  <bench>/_verify/audit_report.csv     one row per slug, with all 4 checks
  <bench>/_verify/audit_report.md      paper-ready table + per-task details
"""
from __future__ import annotations
import csv
import json
import sys
from pathlib import Path

BENCH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "/Users/ofsr/Downloads/NeurIPS_paper_DS_ambiguity/DSBench/evaluate/ambig_ds_metric/benchmark"
)
VDIR = BENCH / "_verify"
MAN = json.load(open(BENCH / "metric_manifest.json"))

rows = []
for jp in sorted(VDIR.glob("*.json")):
    if jp.name.startswith("_"):
        continue
    d = json.load(open(jp))
    slug = d["slug"]
    chks = d.get("checks", {})
    pa = chks.get("plausible_alternatives", {})
    ap = chks.get("ambiguity_preserved", {})
    dr = chks.get("decision_relevant", {})
    tp = chks.get("task_preserved", {})
    rows.append({
        "slug": slug,
        "verdict": d.get("verdict", "?"),
        "true_metric": MAN.get(slug, {}).get("metric_name", "?"),
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

# CSV
csv_p = VDIR / "audit_report.csv"
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

(VDIR / "audit_report.md").write_text("\n".join(md))
print(f"Wrote {csv_p}")
print(f"Wrote {VDIR/'audit_report.md'}")
print(f"Audited {n} tasks: {n_pass} pass, {n-n_pass} fail")
