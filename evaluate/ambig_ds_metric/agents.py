"""Agent adapter: uniform interface around the opencode coding agent.

Returns a 4-tuple expected by the step_*_run_agent*.py scripts:
    (message, tool_uses, iterations, cost)
where:
    message     str   - final assistant text or "ERROR: ..." prefix on failure
    tool_uses   list  - list of tool-call records (best-effort)
    iterations  int   - number of assistant turns
    cost        str   - estimated cost (USD), or "" if unknown

The adapter is intentionally side-effect-free w.r.t. the caller's environment;
it constructs a private subprocess env from the provided api_key/base_url.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# opencode
# ──────────────────────────────────────────────────────────────────────────────

# Minimal config registering a single OpenAI-compatible provider whose baseURL
# and apiKey are taken from environment variables. We declare the user's chosen
# model id explicitly so opencode accepts `--model custom/<id>` without
# needing it to exist in the public models.dev catalog.
_OPENCODE_CONFIG_TEMPLATE = {
    "$schema": "https://opencode.ai/config.json",
    "provider": {
        "custom": {
            "npm": "@ai-sdk/openai-compatible",
            "name": "Custom OpenAI-Compatible Gateway",
            "options": {
                "apiKey": "{env:OPENAI_API_KEY}",
                "baseURL": "{env:OPENAI_BASE_URL}",
            },
            "models": {
                # filled in per-call
            },
        }
    },
}


def _format_cost(total_cost: float, input_tokens: int, output_tokens: int) -> str:
    """Return a cost string, or \"\" if no cost data is available."""
    if total_cost > 0:
        return f"{total_cost:.6f}"
    if input_tokens > 0 or output_tokens > 0:
        # Rough estimate at $3/$15 per 1M tokens (a common mid-tier rate).
        est = input_tokens * 3.0 / 1_000_000 + output_tokens * 15.0 / 1_000_000
        return f"~{est:.6f}"
    return ""


def _write_opencode_config(cwd: Path, model_id: str) -> Path:
    cfg = json.loads(json.dumps(_OPENCODE_CONFIG_TEMPLATE))  # deep copy
    cfg["provider"]["custom"]["models"][model_id] = {
        "name": model_id,
        "tool_call": True,
        "limit": {"context": 200_000, "output": 16_000},
    }
    cfg_path = cwd / "opencode.json"
    cfg_path.write_text(json.dumps(cfg, indent=2))
    return cfg_path


def run_opencode(
    bin_path: str,
    model: str,
    prompt: str,
    cwd: Path,
    api_key: str,
    base_url: str,
    timeout: int = 600,
) -> tuple[str, list, int, str]:
    """Run opencode in `cwd` with a workspace-local config.

    Writes `cwd/opencode.json` registering a custom OpenAI-compatible provider
    that pulls baseURL and apiKey from env, then invokes:
        opencode run --dir <cwd> --model custom/<model>
                     --dangerously-skip-permissions --format json <prompt>
    Parses the JSONL event stream for iterations / tool_uses / cost.
    """
    _write_opencode_config(cwd, model)

    env = os.environ.copy()
    env["OPENAI_API_KEY"] = api_key
    env["OPENAI_BASE_URL"] = base_url
    env["GIT_CEILING_DIRECTORIES"] = str(cwd.parent)

    cmd = [
        bin_path,
        "run",
        "--dir", str(cwd),
        "--model", f"custom/{model}",
        "--dangerously-skip-permissions",
        "--format", "json",
        prompt,
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            cwd=str(cwd), env=env,
        )
        stdout = result.stdout or ""
        stderr = result.stderr or ""
        # Parse JSONL event stream
        message_parts: list[str] = []
        tool_uses: list[dict] = []
        n_assistant_steps = 0
        total_cost = 0.0
        total_input_tokens = 0
        total_output_tokens = 0
        for line in stdout.splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            # Top-level event types observed: step_start, text, tool_use, step_finish.
            # Accept both dash and underscore spellings for forward-compat.
            etype = (ev.get("type") or "").replace("-", "_")
            part = ev.get("part") or {}
            if etype == "step_start":
                n_assistant_steps += 1
            elif etype == "text":
                txt = part.get("text") or ""
                if txt:
                    message_parts.append(txt)
            elif etype in ("tool_use", "tool"):
                state = part.get("state") if isinstance(part.get("state"), dict) else {}
                tool_uses.append({
                    "name": part.get("tool") or part.get("name"),
                    "id": part.get("id"),
                    "state": state.get("status"),
                })
            elif etype == "step_finish":
                c = part.get("cost")
                if isinstance(c, (int, float)):
                    total_cost += float(c)
                # Accumulate token usage for cost estimation when cost is
                # not reported directly (common with gateway providers).
                usage = part.get("usage") or {}
                total_input_tokens += int(usage.get("input_tokens") or
                                          usage.get("prompt_tokens") or 0)
                total_output_tokens += int(usage.get("output_tokens") or
                                           usage.get("completion_tokens") or 0)
        message = "".join(message_parts).strip()
        cost_str = _format_cost(total_cost, total_input_tokens, total_output_tokens)
        if not message and result.returncode != 0:
            return f"ERROR (exit {result.returncode}): {stderr[-500:]}", tool_uses, n_assistant_steps, cost_str
        if not message:
            return f"ERROR: no text in opencode output: {stdout[-300:]}", tool_uses, n_assistant_steps, cost_str
        return message, tool_uses, n_assistant_steps, cost_str
    except subprocess.TimeoutExpired as e:
        partial = (e.stdout or "")[-3000:] if isinstance(e.stdout, str) else ""
        return f"ERROR: timeout\n{partial}", [], 0, ""
    except FileNotFoundError:
        return f"ERROR: opencode binary not found at {bin_path!r}", [], 0, ""
    except Exception as e:
        return f"ERROR: {e}", [], 0, ""


# ──────────────────────────────────────────────────────────────────────────────
# Dispatcher
# ──────────────────────────────────────────────────────────────────────────────

def run_agent(
    bin_path: str,
    model: str,
    prompt: str,
    cwd: Path,
    api_key: str,
    base_url: str,
    timeout: int = 600,
) -> tuple[str, list, int, str]:
    return run_opencode(bin_path, model, prompt, cwd, api_key, base_url, timeout)


def default_bin() -> str:
    candidate = Path.home() / ".npm-global" / "bin" / "opencode"
    if candidate.is_file():
        return str(candidate)
    return "opencode"
