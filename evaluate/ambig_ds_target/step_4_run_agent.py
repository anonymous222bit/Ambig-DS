#!/usr/bin/env python3
"""Step 4 — run an LLM coding agent on the Ambig-DS-T benchmark and grade.

For each task we:
  1. Build a per-task workspace at <bench>/workspaces/<run>/<slug>/
     - data/                  (symlinks of train.csv / test.csv / sample_submission.csv)
     - task.md                (prompt + submission instructions)
     - _meta.json             (workspace provenance)
  2. Run a coding agent (claw or opencode) inside the workspace.
  3. Locate the agent's submission CSV in the workspace.
  4. Grade it via the per-task DSBench evaluator at
     <bench>/release/tasks/<slug>/eval.py against the held-out
     <bench>/data/<slug>/full/test_answer.csv.

Variants
--------
  full          — agent sees the original prompt + full data (train/test/
                  sample_submission with original column names + target).
  ambig_target  — agent sees the target-ambiguous prompt + ambig data
                  (anonymized features + val_1/val_2 candidate columns,
                  no sample_submission).

Outputs land at <bench>/results/<run>/<slug>/:
    _submission.csv     agent's submission (copied)
    _shape.json         submission shape diagnostics
    _grade.json         { "score": float (raw eval.py output),
                          "score_rpg": float in [0,1] (RPG-normalized),
                          "rpg_baseline": float, "rpg_gt": float,
                          "result_path": "..." }
    _traj.json          agent transcript + tool uses + cost
    ../_runlog.jsonl    per-task one-liner log

Prerequisites:
  - Run step_1_setup_benchmark.py first
  - Set OPENAI_API_KEY (or pass --api-key) and OPENAI_BASE_URL if needed
  - claw or opencode on PATH (or pass --agent-bin)

Usage:
    # Full prompts, all tasks, claw
    python step_4_run_agent.py --benchmark-dir ./benchmark \\
        --variant full --model anthropic_claude_haiku_4_5_v1_0

    # Ambig prompts, opencode, subset
    python step_4_run_agent.py --benchmark-dir ./benchmark \\
        --variant ambig_target --model anthropic_claude_haiku_4_5_v1_0 \\
        --agent opencode \\
        --tasks playground-series-s3e17,playground-series-s3e19
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from agents import run_agent as _dispatch_agent, default_bin

SUBMISSION_NAME = "_submission.csv"
VARIANTS = ("full", "ambig_target")


# --------------------------------------------------------------------------- #
def load_tasks(benchmark_dir: Path) -> list[str]:
    return [l.strip() for l in
            (benchmark_dir / "task_list.txt").read_text().splitlines()
            if l.strip()]


# --------------------------------------------------------------------------- #
def build_workspace(slug: str, variant: str, benchmark_dir: Path,
                    workspaces_dir: Path, run_name: str) -> tuple[Path, str]:
    """Build workspaces/<run>/<slug>/ with data symlinks and task.md."""
    if variant == "full":
        src_data = benchmark_dir / "data" / slug / "full"
        prompt_file = benchmark_dir / "release" / "tasks" / slug / "task.txt"
        # Full variant ships train/test/sample_submission to the agent.
        copy_files = ["train.csv", "test.csv", "sample_submission.csv"]
    elif variant == "ambig_target":
        src_data = benchmark_dir / "data" / slug / "ambig"
        prompt_file = benchmark_dir / "release" / "tasks" / slug / "task_ambig.txt"
        # Ambig variant deliberately withholds sample_submission.csv (per the
        # prompt rewrite in step 4 of the create_datasets pipeline).
        copy_files = ["train.csv", "test.csv"]
    else:
        raise ValueError(f"unknown variant: {variant}")

    if not src_data.exists():
        raise FileNotFoundError(f"{variant} data missing for {slug}: {src_data}")
    if not prompt_file.exists():
        raise FileNotFoundError(f"prompt missing: {prompt_file}")

    ws = workspaces_dir / run_name / slug
    if ws.exists():
        shutil.rmtree(ws)
    ws.mkdir(parents=True)
    ws_data = ws / "data"
    ws_data.mkdir()
    for name in copy_files:
        src = src_data / name
        if not src.exists():
            continue  # ambig has no sample_submission, that's fine
        (ws_data / name).symlink_to(src.resolve())

    submission_path = ws / SUBMISSION_NAME
    footer = (
        f"\n\n---\n\n"
        f"## Task instructions\n\n"
        f"You are a data scientist. The dataset for this competition is in "
        f"`./data/`. Build a model and write your predictions to "
        f"`{submission_path}` (absolute path). Do not write anything else "
        f"outside the current working directory."
    )
    prompt_text = prompt_file.read_text() + footer
    (ws / "task.md").write_text(prompt_text)

    (ws / "_meta.json").write_text(json.dumps({
        "slug": slug, "variant": variant,
        "prompt_source": str(prompt_file),
        "data_source": str(src_data),
        "submission_path": str(submission_path),
        "built_at": datetime.now().isoformat(),
    }, indent=2))

    return ws, prompt_text


# --------------------------------------------------------------------------- #
def run_agent(agent: str, bin_path: str, model: str, prompt: str, cwd: Path,
              api_key: str, base_url: str, timeout: int = 600):
    return _dispatch_agent(agent, bin_path, model, prompt, cwd,
                           api_key, base_url, timeout=timeout)


def recover_from_session(workspace: Path) -> dict:
    """Parse claw's session jsonl to recover iteration count + token usage."""
    out = {
        "session_path": None, "n_messages": 0, "n_assistant": 0,
        "n_tool_uses": 0, "input_tokens": 0, "output_tokens": 0,
        "cache_read_tokens": 0, "cache_creation_tokens": 0,
    }
    sess_dir = workspace / ".claw" / "sessions"
    if not sess_dir.exists():
        return out
    sessions = sorted(sess_dir.rglob("session-*.jsonl"))
    if not sessions:
        return out
    sess = sessions[-1]
    out["session_path"] = str(sess)
    for ln in sess.read_text().splitlines():
        try:
            obj = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if obj.get("type") != "message":
            continue
        msg = obj.get("message", {})
        out["n_messages"] += 1
        if msg.get("role") == "assistant":
            out["n_assistant"] += 1
            for blk in msg.get("blocks") or []:
                if isinstance(blk, dict) and blk.get("type") == "tool_use":
                    out["n_tool_uses"] += 1
        usage = msg.get("usage") or {}
        out["input_tokens"]          += int(usage.get("input_tokens", 0) or 0)
        out["output_tokens"]         += int(usage.get("output_tokens", 0) or 0)
        out["cache_read_tokens"]     += int(usage.get("cache_read_input_tokens", 0) or 0)
        out["cache_creation_tokens"] += int(usage.get("cache_creation_input_tokens", 0) or 0)
    return out


# --------------------------------------------------------------------------- #
def find_submission(workspace: Path) -> Path | None:
    for cand in [SUBMISSION_NAME, "submission.csv", "predictions.csv", "preds.csv"]:
        p = workspace / cand
        if p.exists() and p.is_file():
            return p
    csvs = [p for p in workspace.glob("*.csv")
            if p.is_file() and "sample_submission" not in p.name]
    return csvs[0] if len(csvs) == 1 else None


def submission_shape(sub_path: Path, slug: str, benchmark_dir: Path) -> dict:
    out: dict = {"path": str(sub_path)}
    try:
        df = pd.read_csv(sub_path)
    except Exception as e:
        return {**out, "error": f"read failed: {e}"}
    out["n_rows"] = int(len(df))
    out["n_cols"] = int(len(df.columns))
    out["columns"] = list(map(str, df.columns))

    sample_p = benchmark_dir / "data" / slug / "full" / "sample_submission.csv"
    if sample_p.exists():
        try:
            sample = pd.read_csv(sample_p)
            out["sample_n_rows"] = int(len(sample))
            out["sample_n_cols"] = int(len(sample.columns))
            out["matches_sample_shape"] = (
                len(df) == len(sample) and list(df.columns) == list(sample.columns)
            )
        except Exception:
            pass
    return out


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
def _original_target_name(benchmark_dir: Path, slug: str) -> str | None:
    """Look up the original target column name from the per-task manifest."""
    mp = benchmark_dir / "release" / "tasks" / slug / "_manifest.json"
    if not mp.exists():
        return None
    try:
        m = json.loads(mp.read_text())
    except Exception:
        return None
    if "task" in m and isinstance(m["task"], dict):
        return m["task"].get("original_target_name")
    return m.get("original_target_name")


def grade(sub_path: Path, slug: str, benchmark_dir: Path,
          out_dir: Path) -> dict:
    """Run release/tasks/<slug>/eval.py against the full test_answer.csv.

    The DSBench evaluator CLI is:
      python eval.py --answer_file <answers.csv> --predict_file <pred.csv>
                     --path <out_dir> --name <slug>
    It writes a single float to <out_dir>/<slug>/result.txt.

    The ambig-variant agent writes its predictions to a column named
    `prediction` (per the rewritten prompt), but DSBench eval.py expects
    the column named after the *original* target. We materialise an
    aligned copy of the submission with the column renamed before grading.
    """
    eval_py = benchmark_dir / "release" / "tasks" / slug / "eval.py"
    answers = benchmark_dir / "data" / slug / "full" / "test_answer.csv"
    if not eval_py.exists():
        return {"error": f"eval.py missing: {eval_py}"}
    if not answers.exists():
        return {"error": f"test_answer.csv missing: {answers}"}

    # Align submission column to whatever eval.py expects (the original target).
    graded_sub = sub_path
    rename_info = None
    target_name = _original_target_name(benchmark_dir, slug)
    if target_name:
        try:
            df = pd.read_csv(sub_path)
            if target_name not in df.columns and "prediction" in df.columns:
                df = df.rename(columns={"prediction": target_name})
                graded_sub = out_dir / "_submission_for_grader.csv"
                df.to_csv(graded_sub, index=False)
                rename_info = {"prediction": target_name}
        except Exception as e:
            return {"error": f"submission rename failed: {e}"}

    grade_dir = out_dir / "_grade"
    grade_dir.mkdir(parents=True, exist_ok=True)
    # eval.py writes to <grade_dir>/<slug>/result.txt without mkdir-ing the
    # per-slug subdir, so create it ourselves.
    (grade_dir / slug).mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(eval_py),
        "--answer_file", str(answers),
        "--predict_file", str(graded_sub),
        "--path", str(grade_dir),
        "--name", slug,
    ]
    try:
        # stdin=DEVNULL: under nohup/caffeinate the parent's fd 0 may be
        # invalid for grandchildren, causing eval.py to crash at Python
        # startup with `OSError: [Errno 9] Bad file descriptor` from
        # init_sys_streams. Always feed it /dev/null.
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=180,
            stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired:
        return {"error": "eval.py timed out"}
    if proc.returncode != 0:
        return {
            "error": f"eval.py exit {proc.returncode}",
            "stderr": proc.stderr[-500:],
            "stdout": proc.stdout[-500:],
        }

    result_txt = grade_dir / slug / "result.txt"
    if not result_txt.exists():
        return {
            "error": f"result.txt not produced",
            "expected_path": str(result_txt),
            "stderr": proc.stderr[-500:],
        }
    try:
        score = float(result_txt.read_text().strip().splitlines()[0])
    except Exception as e:
        return {"error": f"result.txt unreadable: {e}",
                "result_path": str(result_txt)}
    out = {
        "score": score,
        "result_path": str(result_txt),
        "answer_file": str(answers),
        "submission_file": str(sub_path),
    }
    # RPG normalization (Relative Performance Gap):
    #   max((p - b) / (g - b), 0)
    # where g = best-known (DSBench save_performance/GT/<slug>/result.txt),
    #       b = baseline   (DSBench save_performance/baseline/<slug>/result.txt).
    # Stored in _grade.json so step_6_aggregate.py can macro-average without
    # re-reading per-task baselines.
    bl_dir = benchmark_dir / "baselines" / slug
    gt_p, bl_p = bl_dir / "gt.txt", bl_dir / "baseline.txt"
    if gt_p.exists() and bl_p.exists():
        try:
            g = float(gt_p.read_text().strip().splitlines()[0])
            b = float(bl_p.read_text().strip().splitlines()[0])
            denom = g - b
            if denom != 0:
                rpg = max((score - b) / denom, 0.0)
                out["score_rpg"] = rpg
                out["rpg_baseline"] = b
                out["rpg_gt"] = g
            else:
                out["score_rpg"] = None
                out["rpg_error"] = "g == b (zero denominator)"
        except Exception as e:
            out["score_rpg"] = None
            out["rpg_error"] = f"parse failed: {e}"
    else:
        out["score_rpg"] = None
        out["rpg_error"] = f"baselines missing in {bl_dir}"
    if rename_info:
        out["renamed_submission_file"] = str(graded_sub)
        out["column_renamed"] = rename_info
    return out


# --------------------------------------------------------------------------- #
def run_one(slug: str, variant: str, model: str, args, run_dir: Path,
            benchmark_dir: Path) -> dict:
    log: dict = {"slug": slug, "variant": variant, "model": model,
                 "started_at": datetime.now().isoformat()}

    out_task = run_dir / slug
    out_task.mkdir(parents=True, exist_ok=True)
    grade_file = out_task / "_grade.json"
    if args.skip_existing and grade_file.exists():
        # Only skip if the previous grade was successful. A previous run that
        # failed at the eval-subprocess step (no "score" key, has "error")
        # should be re-graded; otherwise transient infra errors would be
        # frozen forever.
        try:
            prev = json.loads(grade_file.read_text())
            if "score" in prev and prev.get("score") is not None:
                return {**log, "status": "skipped_existing"}
        except Exception:
            pass

    try:
        ws, prompt_text = build_workspace(
            slug, variant, benchmark_dir,
            benchmark_dir / "workspaces", args.run_name)
    except Exception as e:
        return {**log, "status": "build_failed", "error": str(e)}

    if args.dry_run:
        print(f"\n--- DRY RUN: {slug} ({variant}) ---")
        print(f"workspace: {ws}")
        print(f"data files: {sorted(p.name for p in (ws / 'data').iterdir())}")
        print(f"prompt ({len(prompt_text)} chars):")
        print(prompt_text[:1500] + ("..." if len(prompt_text) > 1500 else ""))
        return {**log, "status": "dry_run"}

    t0 = time.time()
    message, tool_uses, iters, cost = run_agent(
        args.agent, args.agent_bin, model, prompt_text, ws,
        args.api_key, args.base_url, timeout=args.timeout)
    elapsed = time.time() - t0

    timed_out = isinstance(message, str) and message.startswith("ERROR: timeout")
    traj = {"slug": slug, "variant": variant, "model": model,
            "elapsed_sec": elapsed, "iterations": iters, "cost": cost,
            "summary": message, "tool_uses": tool_uses, "timed_out": timed_out}
    (out_task / "_traj.json").write_text(json.dumps(traj, indent=2))

    sub_in_ws = find_submission(ws)
    if sub_in_ws is not None:
        sub_dest = out_task / SUBMISSION_NAME
        shutil.copy2(sub_in_ws, sub_dest)
        shape = submission_shape(sub_dest, slug, benchmark_dir)
        (out_task / "_shape.json").write_text(json.dumps(shape, indent=2, default=str))
        report = grade(sub_dest, slug, benchmark_dir, out_task)
    else:
        (out_task / "_shape.json").write_text(json.dumps(
            {"error": "no submission found"}, indent=2))
        report = {"error": "no submission found", "submission_exists": False}

    grade_file.write_text(json.dumps(report, indent=2, default=str))

    log.update({
        "status": "ok" if "error" not in report else "graded_with_error",
        "elapsed_sec": round(elapsed, 1),
        "iterations": iters,
        "score": report.get("score"),
    })
    return log


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--benchmark-dir", type=Path, required=True)
    p.add_argument("--variant", choices=VARIANTS, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--tasks", default="all",
                   help="'all' or comma-separated slugs")
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--agent", choices=["claw", "opencode"], default="claw")
    p.add_argument("--agent-bin", default=None)
    p.add_argument("--api-key", default=None)
    p.add_argument("--base-url",
                   default=os.environ.get("OPENAI_BASE_URL",
                                          "https://api.openai.com/v1"))
    p.add_argument("--run-name", default=None,
                   help="Run dir name (default: <agent>_<model>_<variant>)")
    args = p.parse_args()

    if args.agent_bin is None:
        args.agent_bin = default_bin(args.agent)

    benchmark_dir = args.benchmark_dir.resolve()
    if not (benchmark_dir / "task_list.txt").exists():
        sys.exit(f"task_list.txt not found in {benchmark_dir}. "
                 "Run step_1_setup_benchmark.py first.")

    args.api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not args.api_key and not args.dry_run:
        sys.exit("No API key. Set OPENAI_API_KEY or pass --api-key.")

    args.run_name = args.run_name or f"{args.agent}_{args.model}_{args.variant}"
    results_dir = benchmark_dir / "results" / args.run_name
    results_dir.mkdir(parents=True, exist_ok=True)

    all_tasks = load_tasks(benchmark_dir)
    if args.tasks == "all":
        tasks = all_tasks
    else:
        wanted = {t.strip() for t in args.tasks.split(",")}
        tasks = [t for t in all_tasks if t in wanted]
        missing = wanted - set(tasks)
        if missing:
            print(f"WARNING: unknown tasks: {missing}")

    print(f"Run: {args.run_name}")
    print(f"Tasks: {len(tasks)}, Model: {args.model}, Variant: {args.variant}")
    print(f"Results: {results_dir}\n")

    runlog = results_dir / "_runlog.jsonl"
    with runlog.open("a") as fh:
        for i, slug in enumerate(tasks, 1):
            print(f"[{i}/{len(tasks)}] {slug}...", end=" ", flush=True)
            log = run_one(slug, args.variant, args.model, args,
                          results_dir, benchmark_dir)
            status = log.get("status", "?")
            score = log.get("score")
            score_str = (f"score={score:.4f}"
                         if isinstance(score, (int, float)) else str(score))
            print(f"{status} {score_str}")
            fh.write(json.dumps(log, default=str) + "\n")
            fh.flush()


if __name__ == "__main__":
    main()
