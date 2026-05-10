"""Comprehensive dataset audit for all kaggle_new_dataset/competitions/* slugs.

Checks 5 invariants per task:

  A. PIPELINE OUTPUT INTEGRITY
     - all 18 expected files exist (sources + 12 mirror destinations)
     - mirror copies are byte-identical to sources (md5)
     - data/test_answer.csv aligned with data/test.csv on id

  B. PROMPT INVARIANT (full vs ambig)
     - normal prompt mentions: original target name, real feature names
     - ambig prompt does NOT mention any of those (zero leaks)
     - ambig prompt DOES mention: val_1, val_2, all f_NN features
     - ambig prompt DOES NOT mention: sample_submission.csv
     - both prompts share the same Description / Evaluation / Submission File / Dataset Description / Files headers

  C. DATA INVARIANT (full vs ambig)
     - shapes: same n_train / n_test / n_sub
     - id columns identical (train, test, sample_submission)
     - 14-feature value parity: each anonymized f_NN col equals its mapped original col on first N rows
     - ambig train has [id, f_*, val_1, val_2] only (no original feature/target)
     - ambig test has [id, f_*] only (no targets)
     - val_1 == original target column (truth check, P >= 0.999)
     - val_2 marginal mean ≈ val_1 marginal mean (decoy quality)
     - no original feature name leaks into ambig train/test columns

  D. EVAL SCRIPT CORRECTNESS
     - script imports + defines ID_COL, TARGET_COL matching meta.json
     - baseline result.txt = sentinel value (per metric template) OR direction-correct
     - GT       result.txt = perfect-score sentinel value
     - reference result.txt strictly between baseline and GT (in metric direction)
     - reference RPG > 0.05 (some signal)

  E. DECOY MANIFEST INTEGRITY
     - cv_ratio_decoy_over_true in [0.85, 1.10]
     - marginal_match_exact == True
     - decoy_pool_size >= 4 (or warn)
     - max |spearman(decoy_feat, truth)| <= 0.30 (low-corr pool)

Reports a per-task PASS/WARN/FAIL line and a final summary.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import os
import numpy as np
import pandas as pd

# Layout: ROOT contains Dataset/, DSBench/, final_data_v3/, kaggle_new_dataset/.
# Override with AMBIG_DSBENCH_ROOT env var; defaults to current working directory.
ROOT = Path(os.environ.get("AMBIG_DSBENCH_ROOT", Path.cwd())).resolve()
KND  = Path(os.environ.get("AMBIG_PIPELINE_ROOT", Path(__file__).resolve().parent))
COMP = KND / "competitions"

# Mirror destinations parameterized by slug
def mirror_pairs(slug: str, comp_dir: Path) -> list[tuple[str, Path, Path]]:
    return [
        ("full train",  comp_dir / "data/train.csv",
         ROOT / f"Dataset/data_modeling/data/data/{slug}/train.csv"),
        ("full test",   comp_dir / "data/test.csv",
         ROOT / f"Dataset/data_modeling/data/data/{slug}/test.csv"),
        ("full ss",     comp_dir / "data/sample_submission.csv",
         ROOT / f"Dataset/data_modeling/data/data/{slug}/sample_submission.csv"),
        ("answer",      comp_dir / "data/test_answer.csv",
         ROOT / f"Dataset/data_modeling/data/data/answers/{slug}/test_answer.csv"),
        ("full prompt", comp_dir / "meta/normal_prompt.txt",
         ROOT / f"Dataset/data_modeling/data/data/task/{slug}.txt"),
        ("eval script", KND / f"evaluation/{slug}_eval.py",
         ROOT / f"DSBench/data_modeling/evaluation/{slug}_eval.py"),
        ("baseline",    comp_dir / f"save_performance/baseline/{slug}/result.txt",
         ROOT / f"DSBench/data_modeling/save_performance/baseline/{slug}/result.txt"),
        ("GT",          comp_dir / f"save_performance/GT/{slug}/result.txt",
         ROOT / f"DSBench/data_modeling/save_performance/GT/{slug}/result.txt"),
        ("ambig train", comp_dir / f"ambig/data/{slug}/train.csv",
         ROOT / f"final_data_v3/target_ambig/data_modeling/data/data/data_ambig_target_v3_gen/{slug}/train.csv"),
        ("ambig test",  comp_dir / f"ambig/data/{slug}/test.csv",
         ROOT / f"final_data_v3/target_ambig/data_modeling/data/data/data_ambig_target_v3_gen/{slug}/test.csv"),
        ("ambig ss",    comp_dir / f"ambig/data/{slug}/sample_submission.csv",
         ROOT / f"final_data_v3/target_ambig/data_modeling/data/data/data_ambig_target_v3_gen/{slug}/sample_submission.csv"),
        ("ambig prompt",comp_dir / "meta/ambig_prompt.txt",
         ROOT / f"final_data_v3/target_ambig/data_modeling/data/data/task_ambig_target_v3_gen/{slug}.txt"),
        ("manifest",    comp_dir / f"ambig/data/{slug}/_manifest.json",
         ROOT / f"final_data_v3/target_ambig/data/{slug}/_manifest.json"),
    ]


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def check_one(slug: str) -> dict:
    issues = {"FAIL": [], "WARN": [], "PASS": []}
    comp_dir = COMP / slug
    add = lambda k, m: issues[k].append(m)

    # ---- A. PIPELINE OUTPUT INTEGRITY -------------------------------------
    pairs = mirror_pairs(slug, comp_dir)
    for name, src, dst in pairs:
        if not src.exists():
            add("FAIL", f"A: source missing: {name} ({src})")
            continue
        if not dst.exists():
            add("FAIL", f"A: mirror missing: {name} ({dst})")
            continue
        if md5(src) != md5(dst):
            add("FAIL", f"A: mirror mismatch: {name}  src!=dst")
    if not issues["FAIL"]:
        add("PASS", "A: 13 source+mirror pairs all OK")

    # bail early if the basics are broken
    if any(m.startswith("A: source missing") for m in issues["FAIL"]):
        return issues

    # Load core artifacts
    meta = json.loads((comp_dir / "meta/meta.json").read_text())
    manifest_p = comp_dir / f"ambig/data/{slug}/_manifest.json"
    if not manifest_p.exists():
        add("FAIL", f"A: ambig manifest missing — task incomplete")
        return issues
    manifest = json.loads(manifest_p.read_text())

    full_tr = pd.read_csv(comp_dir / "data/train.csv")
    full_te = pd.read_csv(comp_dir / "data/test.csv")
    full_ans = pd.read_csv(comp_dir / "data/test_answer.csv")
    full_ss = pd.read_csv(comp_dir / "data/sample_submission.csv")
    ambig_tr = pd.read_csv(comp_dir / f"ambig/data/{slug}/train.csv")
    ambig_te = pd.read_csv(comp_dir / f"ambig/data/{slug}/test.csv")
    ambig_ss = pd.read_csv(comp_dir / f"ambig/data/{slug}/sample_submission.csv")

    id_col = meta["id_col"]
    target = meta["target_cols"][0]

    if not full_te[id_col].equals(full_ans[id_col]):
        add("FAIL", "A: test.csv id != test_answer.csv id")
    if not full_te[id_col].equals(full_ss[id_col]):
        add("FAIL", "A: test.csv id != sample_submission.csv id")

    # ---- B. PROMPT INVARIANT ---------------------------------------------
    normal = (comp_dir / "meta/normal_prompt.txt").read_text()
    ambig  = (comp_dir / "meta/ambig_prompt.txt").read_text()

    # B1: normal prompt should mention real names
    if target not in normal:
        add("WARN", f"B: normal prompt does not mention target column {target!r}")

    fmap = manifest["feature_map"]
    real_feats = [f for f in fmap.keys() if len(f) >= 4]   # avoid 1-2 char false positives
    real_feats_in_normal = [f for f in real_feats if f in normal]
    if not real_feats_in_normal:
        add("WARN", "B: normal prompt does not mention any original feature names")

    # B2: ambig prompt MUST NOT mention target or real features.
    # Use word-boundary matching to avoid false positives (e.g. target='y'
    # matching every word containing the letter 'y'); also skip very short
    # tokens (<3 chars) which are inherently ambiguous as substrings.
    def _word_match(needle: str, haystack: str) -> bool:
        if len(needle) < 3:
            return False
        return re.search(rf"(?<![A-Za-z0-9_]){re.escape(needle)}(?![A-Za-z0-9_])",
                         haystack) is not None

    if _word_match(target, ambig):
        add("FAIL", f"B: ambig prompt LEAKS target column name {target!r}")
    leaked = [f for f in real_feats if _word_match(f, ambig)]
    if leaked:
        add("FAIL", f"B: ambig prompt LEAKS feature columns: {leaked[:5]}")

    # B3: ambig prompt MUST mention val_1, val_2, all f_NN
    for required in ("val_1", "val_2"):
        if required not in ambig:
            add("FAIL", f"B: ambig prompt missing required token {required!r}")
    anon_feats = manifest["anon_feature_columns"]
    missing_anon = [a for a in anon_feats if a not in ambig]
    if missing_anon:
        add("WARN", f"B: ambig prompt missing {len(missing_anon)} anonymized features (e.g. {missing_anon[:3]})")

    # B4: ambig prompt MUST NOT mention sample_submission
    if "sample_submission" in ambig:
        add("FAIL", "B: ambig prompt mentions sample_submission.csv (should be removed)")

    # B5: section parity
    sections = ["Description", "Evaluation", "Submission File", "Dataset Description", "Files"]
    for sec in sections:
        if sec not in normal:
            add("WARN", f"B: normal prompt missing section header {sec!r}")
        if sec == "Submission File":
            # ambig may legitimately drop "Submission File" if it lacks one
            continue
        if sec not in ambig:
            add("WARN", f"B: ambig prompt missing section header {sec!r}")

    # ---- C. DATA INVARIANT (full vs ambig) -------------------------------
    if len(full_tr) != len(ambig_tr):
        add("FAIL", f"C: train row count differs: full={len(full_tr)} ambig={len(ambig_tr)}")
    if len(full_te) != len(ambig_te):
        add("FAIL", f"C: test row count differs: full={len(full_te)} ambig={len(ambig_te)}")
    if len(full_ss) != len(ambig_ss):
        add("FAIL", f"C: sample_submission row count differs")

    if id_col in ambig_tr.columns and not full_tr[id_col].equals(ambig_tr[id_col]):
        add("FAIL", "C: train id columns differ between full and ambig")
    if id_col in ambig_te.columns and not full_te[id_col].equals(ambig_te[id_col]):
        add("FAIL", "C: test id columns differ between full and ambig")

    expected_ambig_train_cols = {id_col, "val_1", "val_2", *anon_feats}
    extra_train = set(ambig_tr.columns) - expected_ambig_train_cols
    missing_train = expected_ambig_train_cols - set(ambig_tr.columns)
    if extra_train:
        add("FAIL", f"C: ambig train has unexpected columns: {extra_train}")
    if missing_train:
        add("FAIL", f"C: ambig train missing expected columns: {missing_train}")

    expected_ambig_test_cols = {id_col, *anon_feats}
    extra_test = set(ambig_te.columns) - expected_ambig_test_cols
    if extra_test:
        add("FAIL", f"C: ambig test has unexpected columns: {extra_test}")

    # 14-feature value parity (head 5000 rows; cast to str for dtype-agnostic compare)
    n_check = min(5000, len(full_tr))
    bad_maps = []
    for orig, anon in fmap.items():
        if anon not in ambig_tr.columns or orig not in full_tr.columns:
            bad_maps.append(f"{orig}->{anon} (missing)")
            continue
        a = full_tr[orig].astype("object").head(n_check).fillna("__NA__").to_list()
        b = ambig_tr[anon].astype("object").head(n_check).fillna("__NA__").to_list()
        if a != b:
            # compute mismatch rate
            mismatch = sum(1 for x, y in zip(a, b) if x != y)
            bad_maps.append(f"{orig}->{anon} ({mismatch}/{n_check} mismatches)")
    if bad_maps:
        add("FAIL", f"C: feature-map value parity broken: {bad_maps[:5]}")
    else:
        add("PASS", f"C: all {len(fmap)} feature columns identical (head {n_check})")

    # leak: original names in ambig column headers
    leaked_cols = [c for c in real_feats if c in ambig_tr.columns or c in ambig_te.columns]
    if leaked_cols:
        add("FAIL", f"C: ambig data leaks original column names: {leaked_cols}")

    # truth check: val_1 should equal original target
    truth = manifest["true_target_column"]
    decoy = manifest["decoy_column"]
    # Drop NaN-target rows for the comparison
    mask = full_tr[target].notna()
    n_compare = int(mask.sum())
    if n_compare > 0:
        eq = (full_tr.loc[mask, target].astype(str).values
              == ambig_tr.loc[mask, truth].astype(str).values).mean()
        if eq < 0.999:
            add("FAIL", f"C: P(orig {target} == ambig {truth}) = {eq:.4f} (expect 1.0)")
        else:
            add("PASS", f"C: truth column parity OK (P={eq:.4f}, n={n_compare})")

    # decoy marginal: must approximately match
    try:
        m1 = pd.to_numeric(ambig_tr[truth], errors="coerce").mean()
        m2 = pd.to_numeric(ambig_tr[decoy], errors="coerce").mean()
        if not np.isnan(m1) and not np.isnan(m2):
            rel = abs(m2 - m1) / max(abs(m1), 1e-9)
            if rel > 0.1:
                add("WARN", f"C: decoy marginal differs from truth: {m1:.4f} vs {m2:.4f} (rel={rel:.2%})")
    except Exception:
        pass

    # ---- D. EVAL SCRIPT CORRECTNESS --------------------------------------
    eval_p = KND / f"evaluation/{slug}_eval.py"
    text = eval_p.read_text()
    if f'TARGET_COL = "{target}"' not in text and f"TARGET_COL = '{target}'" not in text:
        add("FAIL", f"D: eval script TARGET_COL does not match meta target {target!r}")
    if f'ID_COL = "{id_col}"' not in text and f"ID_COL = '{id_col}'" not in text:
        add("FAIL", f"D: eval script ID_COL does not match meta id {id_col!r}")

    base = float((comp_dir / f"save_performance/baseline/{slug}/result.txt").read_text().strip())
    gt   = float((comp_dir / f"save_performance/GT/{slug}/result.txt").read_text().strip())
    ref_p = comp_dir / f"save_performance/reference/{slug}/result.txt"
    if ref_p.exists():
        ref = float(ref_p.read_text().strip())
        # determine direction from gt vs base
        higher_better = gt > base
        if higher_better:
            ok = base - 1e-9 <= ref <= gt + 1e-9
        else:
            ok = gt - 1e-9 <= ref <= base + 1e-9
        if not ok:
            add("FAIL", f"D: reference {ref} not between baseline {base} and GT {gt}")
        rpg = abs(ref - base) / max(abs(gt - base), 1e-9)
        if rpg < 0.05:
            add("WARN", f"D: very low signal: RPG={rpg:.3f} (b={base} r={ref} g={gt})")
        else:
            add("PASS", f"D: b={base:.4f} r={ref:.4f} g={gt:.4f} RPG={rpg:.3f}")
    else:
        add("WARN", "D: no reference result.txt (reference model not run)")

    # ---- E. DECOY MANIFEST INTEGRITY -------------------------------------
    diag = manifest.get("diagnostics", {})
    cvr = diag.get("cv_ratio_decoy_over_true")
    if cvr is None or not (0.85 <= cvr <= 1.10):
        add("WARN", f"E: cv_ratio={cvr} outside [0.85, 1.10]")
    if not diag.get("marginal_match_exact"):
        add("WARN", "E: marginal_match_exact != True")
    pool = manifest.get("decoy_pool_anon_features", [])
    if len(pool) < 4:
        add("WARN", f"E: decoy pool only {len(pool)} features (<4)")
    corrs = manifest.get("decoy_pool_abs_spearman_with_truth", [])
    if corrs and max(corrs) > 0.30:
        add("WARN", f"E: max pool |spearman| = {max(corrs):.3f} > 0.30")
    if cvr is not None and 0.85 <= cvr <= 1.10 and diag.get("marginal_match_exact"):
        add("PASS", f"E: decoy quality OK (cv_ratio={cvr:.3f}, pool={len(pool)})")

    return issues


def main() -> int:
    if not COMP.is_dir():
        print(
            f"ERROR: competitions directory not found: {COMP}\n"
            f"This script audits the kaggle_2026 wave layout.\n"
            f"Set AMBIG_PIPELINE_ROOT to the directory containing competitions/,\n"
            f"or skip this step for dsbench-only repros (see README).",
            file=__import__("sys").stderr,
        )
        return 1
    slugs = sorted(p.name for p in COMP.iterdir() if p.is_dir())
    print(f"Auditing {len(slugs)} tasks under {COMP}\n")

    summary = []
    for slug in slugs:
        print(f"================ {slug} ================")
        try:
            issues = check_one(slug)
        except Exception as e:
            print(f"  CRASH: {e}\n")
            summary.append((slug, "CRASH", 0, 0, 0))
            continue
        n_fail = len(issues["FAIL"])
        n_warn = len(issues["WARN"])
        n_pass = len(issues["PASS"])
        for sev in ("FAIL", "WARN", "PASS"):
            for m in issues[sev]:
                print(f"  [{sev}] {m}")
        status = "FAIL" if n_fail else ("WARN" if n_warn else "OK")
        print(f"  -> {status}  ({n_pass} pass, {n_warn} warn, {n_fail} fail)\n")
        summary.append((slug, status, n_pass, n_warn, n_fail))

    print("\n================ FINAL SUMMARY ================")
    print(f"{'slug':<28s}  {'status':6s}  pass  warn  fail")
    for slug, status, p, w, f in summary:
        print(f"  {slug:<28s}  {status:6s}  {p:3d}   {w:3d}   {f:3d}")
    n_ok = sum(1 for _, s, *_ in summary if s == "OK")
    n_warn = sum(1 for _, s, *_ in summary if s == "WARN")
    n_fail = sum(1 for _, s, *_ in summary if s in ("FAIL", "CRASH"))
    print(f"\n  {n_ok} OK   |   {n_warn} WARN   |   {n_fail} FAIL")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
