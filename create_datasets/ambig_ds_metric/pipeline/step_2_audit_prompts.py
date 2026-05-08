#!/usr/bin/env python3
"""Meticulous audit of all ambig_metric.md prompts.

Checks:
1. STRUCTURE: ambig_metric.md exists and is non-empty
2. NEUTRAL SENTENCE: Contains the standard replacement sentence
3. METRIC LEAK: No metric-specific keywords leak into ambig version
4. DIRECTION LEAK: No "minimize"/"maximize"/"higher is better" etc.
5. PROBABILITY LEAK: "probability"/"probabilities" should be neutralized
6. DIFF SANITY: ambig should differ from full (not identical copy)
7. PRESERVATION: Data descriptions, submission columns preserved
8. EVAL SECTION: Evaluation section body should ONLY be the neutral sentence

Usage:
    python step_2_audit_prompts.py --benchmark-dir ./benchmark
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

# ─── Per-task metric keywords ───
METRIC_ALIASES = {
    "log loss": [r"\blog[\s-]*loss\b", r"\blogloss\b", r"\bcross[\s-]*entropy\b"],
    "auc": [r"\bauc\b", r"\barea\s+under\s+(the\s+)?(roc\s+)?curve\b"],
    "roc": [r"\broc\b"],
    "rmse": [r"\brmse\b", r"\broot[\s-]*mean[\s-]*squared?\s+error\b"],
    "mse": [r"\bmse\b", r"\bmean[\s-]*squared?\s+error\b"],
    "mae": [r"\bmae\b", r"\bmean\s+absolute\s+error\b"],
    "accuracy": [r"\baccuracy\b"],
    "f1": [r"\bf1[\s-]*score\b", r"\bf1\b", r"\bf[\s-]*measure\b"],
    "f0.5": [r"\bf0\.5\b"],
    "precision": [r"\bprecision\b"],
    "recall": [r"\brecall\b"],
    "map": [r"\bmAP\b", r"\bmean\s+average\s+precision\b"],
    "map@": [r"\bmap@\d+\b", r"\bMAP@\d+\b"],
    "dice": [r"\bdice\b", r"\bsorensen\b"],
    "iou": [r"\biou\b", r"\bintersection[\s-]*over[\s-]*union\b", r"\bjaccard\b"],
    "jaccard": [r"\bjaccard\b"],
    "kappa": [r"\bkappa\b", r"\bqwk\b", r"\bquadratic\s+weighted\s+kappa\b"],
    "spearman": [r"\bspearman\b"],
    "kendall": [r"\bkendall\b", r"\bkendall[\s-]*tau\b"],
    "levenshtein": [r"\blevenshtein\b", r"\bedit[\s-]*distance\b"],
    "hausdorff": [r"\bhausdorff\b"],
    "kullback": [r"\bkullback\b", r"\bkl[\s-]*divergence\b"],
    "correlation": [r"\bcorrelation\s+coefficient\b"],
    "lwlrap": [r"\blwlrap\b", r"\blabel[\s-]*ranking\s+average\s+precision\b"],
    "mcrmse": [r"\bmcrmse\b", r"\bmean\s+column[\s-]*wise\s+r?mse\b"],
    "crps": [r"\bcrps\b"],
    "pf1": [r"\bpf1\b", r"\bprobabilistic\s+f1\b"],
}

DIRECTION_PATTERNS = [
    r"\bminimize\b",
    r"\bmaximize\b",
    r"\bhigher[\s-]*(is|scores?\s+are)\s+better\b",
    r"\blower[\s-]*(is|scores?\s+are)\s+better\b",
    r"\bhigh(?:er)?\s+score\s+(?:is|means)\b",
    r"\blow(?:er)?\s+score\s+(?:is|means)\b",
    r"\boptimize\b",
    r"\bpenalizes?\b",
]

PROB_PATTERNS = [
    r"\bpredict\s+(?:a\s+)?probabilit(?:y|ies)\b",
    r"\bsubmit\s+(?:a\s+)?probabilit(?:y|ies)\b",
    r"\bpredicted\s+probabilit(?:y|ies)\b",
    r"\bclass\s+probabilit(?:y|ies)\b",
    r"\bprobability\s+(?:that|of|for|between|from|column|value)\b",
]

NEUTRAL_SENTENCE = "Submissions are scored against the held-out ground truth on this task."

FALSE_POSITIVES = {
    "roc": [
        r"\bprocess\b", r"\bprocedure\b", r"\bproceed\b", r"\bprocessing\b",
        r"\bprocurable\b", r"\brock\b", r"\bbroccoli\b", r"\bcroc\b",
    ],
    "iou": [
        r"\bvarious\b", r"\bprevious\b", r"\bserious\b", r"\bcurious\b",
        r"\bobvious\b", r"\bdelicious\b", r"\bfurious\b",
    ],
    "map": [
        r"\bmap\b(?!@)",
        r"\bmaps\b", r"\bmapping\b", r"\bmapped\b", r"\broadmap\b", r"\bheatmap\b",
    ],
    "f1": [r"\b[0-9a-f]{6,}\b"],
    "precision": [
        r"\bfloating[\s-]*point\s+precision\b", r"\bdouble[\s-]*precision\b",
        r"\bhalf[\s-]*precision\b", r"\bfull[\s-]*precision\b",
    ],
    "recall": [],
    "accuracy": [
        r"\baccurate\b", r"\baccurately\b", r"\binaccurate\b", r"\binaccuracy\b",
    ],
    "correlation": [
        r"\bcorrelat(?:ed|ing|ion)\b.*(?:feature|column|variable|data|between|among)",
    ],
}


def get_metric_keywords(slug: str, manifest: dict) -> list[tuple[str, re.Pattern]]:
    """Build per-task metric leak patterns from the manifest's metric_name.

    Uses only curated METRIC_ALIASES. A previous version also extracted every
    literal word ≥4 chars from `metric_name`, but `metric_name` is a free-form
    description that often includes generic words ("bias", "with", "over",
    "columns", "error", "species", "jigsaw", "unintended", ...). Those
    produced overwhelming false positives, so literal extraction is dropped.
    """
    entry = manifest.get(slug, {})
    metric_name = entry.get("metric_name", "").lower()

    keywords = []
    for key, patterns in METRIC_ALIASES.items():
        if key in metric_name:
            for p in patterns:
                keywords.append((f"metric:{key}", re.compile(p, re.IGNORECASE)))
    return keywords


def is_false_positive(key: str, match_text: str, line: str) -> bool:
    base_key = key.split(":")[1] if ":" in key else key
    for fp_key, fp_patterns in FALSE_POSITIVES.items():
        if fp_key == base_key:
            for fp in fp_patterns:
                if re.search(fp, line, re.IGNORECASE):
                    return True
    if base_key in ("f1", "f0.5"):
        context = line[max(0, line.lower().find(match_text.lower())-5):
                       line.lower().find(match_text.lower())+len(match_text)+5]
        if re.match(r"^[0-9a-f]+$", context.strip(), re.IGNORECASE):
            return True
    return False


def find_eval_section(text: str) -> tuple[int, int, str] | None:
    lines = text.split("\n")
    eval_start = None
    eval_header = None

    for i, line in enumerate(lines):
        if re.match(r"^#{1,4}\s+(evaluation|metric|scoring)\b", line, re.IGNORECASE):
            eval_start = i
            eval_header = line.strip()
            break

    if eval_start is None:
        return None

    # Treat the eval section as ending at the NEXT header of any level so a
    # following `### Submission File` subsection is not counted as eval body.
    eval_end = len(lines)
    for i in range(eval_start + 1, len(lines)):
        if re.match(r"^#+\s+", lines[i]):
            eval_end = i
            break

    return eval_start, eval_end, eval_header


def audit_task(slug: str, prompts_dir: Path, manifest: dict) -> list[dict]:
    issues = []
    slug_dir = prompts_dir / slug

    full_path = slug_dir / "full.md"
    ambig_path = slug_dir / "ambig_metric.md"

    if not full_path.exists():
        issues.append({"severity": "CRITICAL", "check": "existence", "msg": "full.md missing"})
        return issues
    if not ambig_path.exists():
        issues.append({"severity": "CRITICAL", "check": "existence", "msg": "ambig_metric.md missing"})
        return issues

    full_text = full_path.read_text()
    ambig_text = ambig_path.read_text()

    if len(ambig_text.strip()) < 50:
        issues.append({"severity": "CRITICAL", "check": "empty", "msg": f"ambig_metric.md too short ({len(ambig_text)} chars)"})
        return issues

    if full_text.strip() == ambig_text.strip():
        issues.append({"severity": "CRITICAL", "check": "identical", "msg": "ambig_metric.md is identical to full.md"})
        return issues

    if NEUTRAL_SENTENCE not in ambig_text:
        close = re.search(r"submissions?\s+are\s+scored\s+against", ambig_text, re.IGNORECASE)
        if close:
            issues.append({"severity": "WARNING", "check": "neutral_sentence",
                          "msg": f"Variant of neutral sentence found: '{close.group()[:80]}...'"})
        else:
            issues.append({"severity": "CRITICAL", "check": "neutral_sentence",
                          "msg": "Standard neutral sentence NOT found in ambig_metric.md"})

    eval_info = find_eval_section(ambig_text)
    if eval_info:
        start, end, header = eval_info
        lines = ambig_text.split("\n")
        body_lines = [l.strip() for l in lines[start+1:end] if l.strip()]
        body_text = " ".join(body_lines)
        if NEUTRAL_SENTENCE not in body_text and len(body_lines) > 0:
            if len(body_text) > 100:
                issues.append({"severity": "CRITICAL", "check": "eval_section",
                              "msg": f"Eval section has extra content ({len(body_lines)} lines): {body_text[:150]}..."})

    metric_keywords = get_metric_keywords(slug, manifest)
    for label, pattern in metric_keywords:
        for match in pattern.finditer(ambig_text):
            matched_text = match.group()
            line_start = ambig_text.rfind("\n", 0, match.start()) + 1
            line_end = ambig_text.find("\n", match.end())
            if line_end == -1:
                line_end = len(ambig_text)
            line = ambig_text[line_start:line_end].strip()

            if is_false_positive(label, matched_text, line):
                continue

            data_context = re.search(
                r"(column|field|feature|attribute|variable|header|row|record)\b",
                line, re.IGNORECASE
            )
            if data_context and label.startswith("literal:"):
                continue

            issues.append({"severity": "CRITICAL", "check": "metric_leak",
                          "msg": f"Metric keyword [{label}] found: '{matched_text}' in: {line[:120]}"})

    # Direction-keyword leaks are only meaningful inside the Evaluation
    # section. Outside of it, words like "optimize algorithms" or "minimize
    # unintended bias" are unrelated to the metric direction.
    if eval_info:
        e_start, e_end, _ = eval_info
        eval_lines = ambig_text.split("\n")[e_start:e_end]
        eval_body_text = "\n".join(eval_lines)
        for dp in DIRECTION_PATTERNS:
            for match in re.finditer(dp, eval_body_text, re.IGNORECASE):
                line_start = eval_body_text.rfind("\n", 0, match.start()) + 1
                line_end = eval_body_text.find("\n", match.end())
                if line_end == -1:
                    line_end = len(eval_body_text)
                line = eval_body_text[line_start:line_end].strip()
                issues.append({"severity": "CRITICAL", "check": "direction_leak",
                              "msg": f"Direction keyword '{match.group()}' in eval section: {line[:120]}"})

    for pp in PROB_PATTERNS:
        for match in re.finditer(pp, ambig_text, re.IGNORECASE):
            line_start = ambig_text.rfind("\n", 0, match.start()) + 1
            line_end = ambig_text.find("\n", match.end())
            if line_end == -1:
                line_end = len(ambig_text)
            line = ambig_text[line_start:line_end].strip()
            issues.append({"severity": "WARNING", "check": "probability_leak",
                          "msg": f"Probability hint: '{match.group()}' in: {line[:120]}"})

    if eval_info:
        start, end, header = eval_info
        lines = ambig_text.split("\n")
        body_lines = [l.strip() for l in lines[start+1:end] if l.strip()]
        if len(body_lines) > 2:
            body_preview = " ".join(body_lines[:3])
            issues.append({"severity": "WARNING", "check": "verbose_eval",
                          "msg": f"Eval section has {len(body_lines)} lines (expected 1): {body_preview[:150]}..."})

    ratio = len(ambig_text) / max(len(full_text), 1)
    if ratio < 0.5:
        issues.append({"severity": "WARNING", "check": "length",
                      "msg": f"Ambig is {ratio:.0%} of full length — may have lost content"})
    if ratio > 1.1:
        issues.append({"severity": "WARNING", "check": "length",
                      "msg": f"Ambig is {ratio:.0%} of full length — may have added content"})

    formula_patterns = [
        (r"\$.*(?:log|ln|exp|sqrt|sum|prod|frac|sigma|mu).*\$", "LaTeX formula"),
        (r"(?:=\s*)?-?\s*\d*\s*[/×·]\s*(?:Σ|∑|sum|log|ln)", "Math formula"),
        (r"\bTP\s*[+/]\s*(?:FP|FN|TN)\b", "Confusion matrix formula"),
        (r"\b(?:true|false)\s+(?:positive|negative)\s+rate\b", "TPR/FPR mention"),
    ]
    for fp, label in formula_patterns:
        for match in re.finditer(fp, ambig_text, re.IGNORECASE):
            if match.group() in full_text:
                full_eval = find_eval_section(full_text)
                if full_eval:
                    full_lines = full_text.split("\n")
                    eval_body = "\n".join(full_lines[full_eval[0]:full_eval[1]])
                    if match.group() in eval_body:
                        line_start = ambig_text.rfind("\n", 0, match.start()) + 1
                        line_end = ambig_text.find("\n", match.end())
                        if line_end == -1:
                            line_end = len(ambig_text)
                        issues.append({"severity": "WARNING", "check": "formula",
                                      "msg": f"{label}: {match.group()[:80]} (was in eval section of full)"})

    return issues


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark-dir", type=Path, required=True,
                    help="Benchmark directory with prompts/")
    args = ap.parse_args()

    benchmark_dir = args.benchmark_dir.resolve()
    prompts_dir = benchmark_dir / "prompts"

    task_list_path = benchmark_dir / "task_list.txt"
    if not task_list_path.exists():
        sys.exit(f"task_list.txt not found in {benchmark_dir}")
    task_list = task_list_path.read_text().strip().splitlines()

    manifest_path = prompts_dir / "_metric_manifest.json"
    if not manifest_path.exists():
        manifest_path = benchmark_dir / "metric_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    print(f"Auditing {len(task_list)} tasks...\n")

    all_issues = {}
    critical_count = 0
    warning_count = 0

    for slug in sorted(task_list):
        issues = audit_task(slug, prompts_dir, manifest)
        if issues:
            all_issues[slug] = issues
            for iss in issues:
                if iss["severity"] == "CRITICAL":
                    critical_count += 1
                else:
                    warning_count += 1

    clean = len(task_list) - len(all_issues)
    print(f"{'='*80}")
    print(f"AUDIT SUMMARY: {len(task_list)} tasks")
    print(f"  Clean:    {clean}")
    print(f"  Flagged:  {len(all_issues)} tasks")
    print(f"  CRITICAL: {critical_count}")
    print(f"  WARNING:  {warning_count}")
    print(f"{'='*80}\n")

    if not all_issues:
        print("ALL CLEAN — no issues found.")
        return

    for slug in sorted(all_issues.keys()):
        issues = all_issues[slug]
        crits = [i for i in issues if i["severity"] == "CRITICAL"]
        warns = [i for i in issues if i["severity"] == "WARNING"]

        marker = "CRITICAL" if crits else "WARNING"
        print(f"\n[{marker}] {slug}")
        entry = manifest.get(slug, {})
        print(f"   Metric: {entry.get('metric_name', '?')} (lower_better={entry.get('is_lower_better', '?')})")
        for iss in crits:
            print(f"   CRITICAL [{iss['check']}]: {iss['msg']}")
        for iss in warns:
            print(f"   WARNING  [{iss['check']}]: {iss['msg']}")

    print(f"\n{'='*80}")
    print("ISSUE TYPE BREAKDOWN:")
    type_counts = Counter()
    for issues in all_issues.values():
        for iss in issues:
            type_counts[f"{iss['severity']}:{iss['check']}"] += 1
    for k, v in type_counts.most_common():
        print(f"  {k}: {v}")

    sys.exit(1 if critical_count > 0 else 0)


if __name__ == "__main__":
    main()
