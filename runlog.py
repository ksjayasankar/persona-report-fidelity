"""Structured experiment logging + cost tracking.

Folder layout (all machine-regenerable, nothing hand-maintained):
  logs/events-<YYYYMMDD>.jsonl   one JSON line per event (calls, runs, errors)
  logs/costs.json                cumulative token/cost ledger per model
  logs/config-snapshot-*.json    frozen config at each grid launch
  runs/raw/                      authoritative raw run records (harness)
  runs/verification/             raw API verification responses
  runs/scored/                   scoring output (scores.jsonl, exposure.jsonl)
  runs/report/                   tables.md + figures/*.png (report builder)
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any

LOG_DIR = "logs"
_LOCK = threading.Lock()

# $/M tokens (input, output) — synced 2026-08-14; used for the ledger only,
# never for reported results.
PRICES: dict[str, tuple[float, float]] = {
    "openai/gpt-oss-120b": (0.15, 0.60),
    "MiniMaxAI/MiniMax-M3": (0.30, 1.20),
    "deepseek-ai/DeepSeek-V4-Flash-0731": (0.14, 0.28),
    "Qwen/Qwen3.7-Plus": (0.32, 1.28),
    "moonshotai/Kimi-K2.6": (1.20, 4.50),
    "Qwen/Qwen2.5-7B-Instruct-Turbo": (0.30, 0.30),
    "claude-haiku-4-5-20251001": (1.00, 5.00),
    "claude-sonnet-5": (2.00, 10.00),
    "openai/gpt-5.1": (1.25, 10.00),
}


def _today() -> str:
    return time.strftime("%Y%m%d")


def log_event(event: str, **fields: Any) -> None:
    rec = {"t": round(time.time(), 3), "event": event, **fields}
    os.makedirs(LOG_DIR, exist_ok=True)
    with _LOCK:
        with open(os.path.join(LOG_DIR, f"events-{_today()}.jsonl"), "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")


def usage_of(result: dict[str, Any]) -> tuple[int, int]:
    """(input_tokens, output_tokens) from a providers.py result, 0s if absent."""
    raw = result.get("raw") or {}
    u = raw.get("usage") or {}
    if "prompt_tokens" in u:      # OpenAI-style (Together, OpenRouter)
        return int(u.get("prompt_tokens") or 0), int(u.get("completion_tokens") or 0)
    if "input_tokens" in u:       # Anthropic
        return int(u.get("input_tokens") or 0), int(u.get("output_tokens") or 0)
    return 0, 0


def add_cost(model: str, result: dict[str, Any]) -> None:
    tin, tout = usage_of(result)
    if not (tin or tout):
        return
    os.makedirs(LOG_DIR, exist_ok=True)
    path = os.path.join(LOG_DIR, "costs.json")
    with _LOCK:
        ledger: dict[str, Any] = {}
        if os.path.exists(path):
            with open(path) as f:
                ledger = json.load(f)
        row = ledger.setdefault(model, {"calls": 0, "in": 0, "out": 0, "usd": 0.0})
        row["calls"] += 1
        row["in"] += tin
        row["out"] += tout
        pin, pout = PRICES.get(model, (0.0, 0.0))
        row["usd"] = round(row["in"] / 1e6 * pin + row["out"] / 1e6 * pout, 4)
        ledger["_total_usd"] = round(
            sum(v["usd"] for k, v in ledger.items() if k != "_total_usd"), 4)
        with open(path, "w") as f:
            json.dump(ledger, f, indent=1)


def snapshot_config(tag: str, config: dict[str, Any]) -> str:
    os.makedirs(LOG_DIR, exist_ok=True)
    path = os.path.join(LOG_DIR, f"config-snapshot-{tag}-{int(time.time())}.json")
    with open(path, "w") as f:
        json.dump(config, f, indent=1, default=str)
    return path
