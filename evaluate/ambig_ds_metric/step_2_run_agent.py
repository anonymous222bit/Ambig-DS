#!/usr/bin/env python3
"""Run the Ambig-DS-M benchmark: build workspace, run agent, grade submission.

This is the main entry point for running experiments. It:
  1. Builds a workspace per task (symlinks data, creates task.md from prompt)
  2. Runs an LLM-powered coding agent (claw or opencode) in the workspace
  3. Locates the agent's submission CSV
  4. Grades it against the held-out ground truth via MLE-bench

Prerequisites:
  - Run step_1_setup_benchmark.py first to download prompts + data
  - Set OPENAI_API_KEY (or pass --api-key)
  - Agent binary on PATH or pass --agent-bin

Usage:
    # Full prompts, all tasks
    python step_2_run_agent.py --benchmark-dir ./benchmark --variant full --model gpt-4o

    # Ambiguous prompts, specific tasks
    python step_2_run_agent.py --benchmark-dir ./benchmark --variant ambig_metric --model gpt-4o \\
        --tasks aerial-cactus-identification,dog-breed-identification

    # Dry run (build workspace, print prompt, don't call agent)
    python step_2_run_agent.py --benchmark-dir ./benchmark --variant full --model gpt-4o --dry-run
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


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_tasks(benchmark_dir: Path) -> list[str]:
    return [l.strip() for l in (benchmark_dir / "task_list.txt").read_text().splitlines() if l.strip()]


def get_registry(data_dir: Path):
    """Get MLE-bench registry pointed at our data directory."""
    from mlebench.registry import registry as _reg
    return _reg.set_data_dir(data_dir)


# ──────────────────────────────────────────────────────────────────────────────
# Workspace builder
# ──────────────────────────────────────────────────────────────────────────────

def build_workspace(
    slug: str,
    variant: str,
    benchmark_dir: Path,
    workspaces_dir: Path,
    run_name: str,
) -> tuple[Path, str]:
    """Build workspaces/<run>/<slug>/ with data symlinks and task.md.

    Returns (workspace_path, prompt_text).
    """
    data_dir = benchmark_dir / "data"
    prompts_dir = benchmark_dir / "prompts"

    src_public = data_dir / slug / "prepared" / "public"
    if not src_public.exists():
        raise FileNotFoundError(f"prepared/public missing for {slug}: {src_public}")

    prompt_file = prompts_dir / slug / f"{variant}.md"
    if not prompt_file.exists():
        raise FileNotFoundError(f"prompt missing: {prompt_file}")

    ws = workspaces_dir / run_name / slug
    if ws.exists():
        shutil.rmtree(ws)
    ws.mkdir(parents=True)
    ws_data = ws / "data"
    ws_data.mkdir()

    # Symlink each top-level entry; extract .7z on the fly
    for entry in sorted(src_public.iterdir()):
        if entry.suffix.lower() == ".7z":
            import py7zr
            with py7zr.SevenZipFile(entry, mode="r") as z:
                z.extractall(path=ws_data)
        elif entry.name == "description.md":
            continue  # agent reads task.md, not description.md
        else:
            (ws_data / entry.name).symlink_to(entry.resolve())

    # task.md = prompt + submission instructions
    footer = (
        f"\n\n---\n\n"
        f"## Task instructions\n\n"
        f"You are a data scientist. The dataset for this competition is in `./data/`.\n"
        f"Build a model and write your predictions to `./{SUBMISSION_NAME}` "
        f"(in the current working directory). Do not write anything else outside "
        f"the current working directory."
    )
    prompt_text = prompt_file.read_text() + footer
    (ws / "task.md").write_text(prompt_text)

    (ws / "_meta.json").write_text(json.dumps({
        "slug": slug, "variant": variant,
        "prompt_source": str(prompt_file),
        "data_source": str(src_public),
        "built_at": datetime.now().isoformat(),
    }, indent=2))

    return ws, prompt_text


# ──────────────────────────────────────────────────────────────────────────────
# Agent runner
# ──────────────────────────────────────────────────────────────────────────────

def run_agent(agent: str, bin_path: str, model: str, prompt: str, cwd: Path,
              api_key: str, base_url: str, timeout: int = 600):
    """Dispatch to the configured agent adapter (claw or opencode).

    Returns (message, tool_uses, iterations, cost).
    """
    return _dispatch_agent(agent, bin_path, model, prompt, cwd,
                           api_key, base_url, timeout=timeout)


# ──────────────────────────────────────────────────────────────────────────────
# Session recovery (when the agent times out)
# ──────────────────────────────────────────────────────────────────────────────

def recover_from_session(workspace: Path) -> dict:
    """Parse claw's session JSONL to recover iteration count + token usage.

    Only applicable when --agent claw; opencode does not write .claw/ sessions.
    """
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


# ──────────────────────────────────────────────────────────────────────────────
# Submission discovery + grading
# ──────────────────────────────────────────────────────────────────────────────

def find_submission(workspace: Path) -> Path | None:
    """Locate the agent's submission CSV."""
    for cand in [SUBMISSION_NAME, "submission.csv", "predictions.csv", "preds.csv"]:
        p = workspace / cand
        if p.exists() and p.is_file():
            return p
    csvs = [
        p for p in workspace.glob("*.csv")
        if p.is_file() and "sample_submission" not in p.name
    ]
    return csvs[0] if len(csvs) == 1 else None


def submission_shape(sub_path: Path, slug: str, registry) -> dict:
    """Compute diagnostics about the submitted CSV."""
    out: dict = {"path": str(sub_path)}
    try:
        df = pd.read_csv(sub_path)
    except Exception as e:
        return {**out, "error": f"read failed: {e}"}

    out["n_rows"] = int(len(df))
    out["n_cols"] = int(len(df.columns))
    out["columns"] = list(map(str, df.columns))

    try:
        comp = registry.get_competition(slug)
        sample = pd.read_csv(comp.sample_submission)
        out["sample_n_rows"] = int(len(sample))
        out["sample_n_cols"] = int(len(sample.columns))
        out["matches_sample_shape"] = (
            len(df) == len(sample) and list(df.columns) == list(sample.columns)
        )
    except Exception:
        pass

    return out


def grade(sub_path: Path, slug: str, registry) -> dict:
    """Grade a submission CSV against held-out ground truth."""
    from mlebench.grade import grade_csv

    try:
        comp = registry.get_competition(slug)
        report = grade_csv(sub_path, comp)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}

    out = {}
    for k, v in report.__dict__.items():
        out[k] = v.isoformat() if isinstance(v, datetime) else v
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Per-task driver
# ──────────────────────────────────────────────────────────────────────────────

def run_one(slug: str, variant: str, model: str, args, run_dir: Path,
            benchmark_dir: Path, registry) -> dict:
    """Run one task end-to-end: build -> agent -> find submission -> grade."""
    log: dict = {
        "slug": slug, "variant": variant, "model": model,
        "started_at": datetime.now().isoformat(),
    }

    out_task = run_dir / slug
    out_task.mkdir(parents=True, exist_ok=True)
    grade_file = out_task / "_grade.json"

    if args.skip_existing and grade_file.exists():
        return {**log, "status": "skipped_existing"}

    try:
        ws, prompt_text = build_workspace(
            slug, variant, benchmark_dir,
            benchmark_dir / "workspaces", args.run_name,
        )
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
        args.api_key, args.base_url, timeout=args.timeout,
    )
    elapsed = time.time() - t0

    timed_out = isinstance(message, str) and message.startswith("ERROR: timeout")
    recovered = None
    if timed_out or (iters == 0 and not tool_uses):
        recovered = recover_from_session(ws)
        if recovered.get("n_assistant", 0) > 0:
            iters = recovered["n_assistant"]

    traj = {
        "slug": slug, "variant": variant, "model": model,
        "elapsed_sec": elapsed, "iterations": iters, "cost": cost,
        "summary": message, "tool_uses": tool_uses,
        "timed_out": timed_out,
    }
    if recovered is not None:
        traj["recovered_from_session"] = recovered
    (out_task / "_traj.json").write_text(json.dumps(traj, indent=2))

    sub_in_ws = find_submission(ws)
    sub_dest = out_task / "_submission.csv"
    if sub_in_ws is not None:
        shutil.copy2(sub_in_ws, sub_dest)
        shape = submission_shape(sub_dest, slug, registry)
        (out_task / "_shape.json").write_text(json.dumps(shape, indent=2, default=str))
        report = grade(sub_dest, slug, registry)
    else:
        shape = {"error": "no submission found"}
        (out_task / "_shape.json").write_text(json.dumps(shape, indent=2))
        report = {"error": "no submission found", "submission_exists": False}

    grade_file.write_text(json.dumps(report, indent=2, default=str))

    log.update({
        "status": "ok" if "error" not in report else "graded_with_error",
        "elapsed_sec": round(elapsed, 1),
        "iterations": iters,
        "score": report.get("score"),
        "valid_submission": report.get("valid_submission"),
        "above_median": report.get("above_median"),
        "any_medal": report.get("any_medal"),
    })
    return log


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--benchmark-dir", type=Path, required=True,
                   help="Benchmark directory (created by step_1_setup_benchmark.py)")
    p.add_argument("--variant", choices=["full", "ambig_metric"], required=True)
    p.add_argument("--model", required=True, help="Model name (e.g., gpt-4o)")
    p.add_argument("--tasks", default="all",
                   help="'all' or comma-separated slugs")
    p.add_argument("--timeout", type=int, default=600,
                   help="Seconds per task (default: 600)")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip tasks with existing _grade.json")
    p.add_argument("--dry-run", action="store_true",
                   help="Build workspace only, don't run agent")
    p.add_argument("--agent", choices=["claw", "opencode"], default="claw",
                   help="Coding agent to run (default: claw)")
    p.add_argument("--agent-bin", default=None,
                   help="Path to agent binary (default: 'claw' or 'opencode' on PATH)")
    # Back-compat: --claw-bin still accepted as an alias for --agent-bin.
    p.add_argument("--claw-bin", dest="claw_bin", default=None,
                   help="DEPRECATED alias for --agent-bin")
    p.add_argument("--api-key", default=None,
                   help="OpenAI API key (default: $OPENAI_API_KEY)")
    p.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                   help="OpenAI-compatible API base URL (default: $OPENAI_BASE_URL or OpenAI)")
    p.add_argument("--run-name", default=None,
                   help="Run directory name (default: <agent>_<model>_<variant>)")
    args = p.parse_args()

    if args.agent_bin is None:
        args.agent_bin = args.claw_bin or default_bin(args.agent)

    benchmark_dir = args.benchmark_dir.resolve()
    if not (benchmark_dir / "task_list.txt").exists():
        sys.exit(f"task_list.txt not found in {benchmark_dir}. Run step_1_setup_benchmark.py first.")

    args.api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not args.api_key and not args.dry_run:
        sys.exit("No API key. Set OPENAI_API_KEY or pass --api-key.")

    args.run_name = args.run_name or f"{args.agent}_{args.model}_{args.variant}"
    results_dir = benchmark_dir / "results" / args.run_name
    results_dir.mkdir(parents=True, exist_ok=True)

    registry = get_registry(benchmark_dir / "data")

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
    for i, slug in enumerate(tasks, 1):
        print(f"[{i}/{len(tasks)}] {slug}...", end=" ", flush=True)
        log = run_one(slug, args.variant, args.model, args,
                      results_dir, benchmark_dir, registry)
        status = log.get("status", "?")
        score = log.get("score")
        score_str = f"score={score:.4f}" if isinstance(score, (int, float)) else str(score)
        print(f"{status} {score_str}")

        with open(runlog, "a") as f:
            f.write(json.dumps(log, default=str) + "\n")

    print(f"\n{'='*60}")
    print(f"Done. Results in {results_dir}")
    print(f"Run log: {runlog}")


if __name__ == "__main__":
    main()
