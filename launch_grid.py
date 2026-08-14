"""Grid launcher: bounded per-provider concurrency, resumable, fully logged.

Usage:
  python3 launch_grid.py --dry-run          # counts + cost projection only
  python3 launch_grid.py --pilot            # pilot slice
  python3 launch_grid.py                    # full confirmatory grid
  python3 launch_grid.py --include-tier2    # + condition E cells

Safe to kill and rerun: completed runs are skipped via idempotent raw files.
Progress: logs/manifest-status.json (refreshed every 25 completions) +
logs/events-*.jsonl. Costs: logs/costs.json.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import random
import threading
import time

import harness
import runlog
from items_draft import ITEMS

OPEN_MODELS = [
    {"model": "openai/gpt-oss-120b", "provider": "together"},
    {"model": "MiniMaxAI/MiniMax-M3", "provider": "together"},
    {"model": "deepseek-ai/DeepSeek-V4-Flash-0731", "provider": "together"},
    {"model": "Qwen/Qwen3.7-Plus", "provider": "together"},
]
FRONTIER_MODELS = [
    {"model": "claude-haiku-4-5-20251001", "provider": "anthropic"},
    {"model": "openai/gpt-5.1", "provider": "openrouter"},
]
CONDITIONS_T1 = ["A", "B", "C", "D"]
CONDITIONS_T2 = ["A", "B", "C", "D", "E"]
# Dropped by the pre-registered degenerate-item rule (prepass 2026-08-14,
# runs/prepass/prepass_summary.json): base rate 1.00 in the stated pre-pass.
DROPPED_ITEMS = {"continue_vs_exit_monotony", "assigned_vs_autonomous"}
OPEN_FRAMINGS, OPEN_REPS = [0, 1, 2], 10
FRONTIER_FRAMINGS, FRONTIER_REPS = [0], 5
MAX_CONCURRENCY = {"together": 6, "anthropic": 4, "openrouter": 4}

_done, _failed = 0, 0
_lock = threading.Lock()


def build_full_manifest(conditions: list[str], pilot: bool = False) -> list[dict]:
    item_ids = [i["id"] for i in ITEMS if i["id"] not in DROPPED_ITEMS]
    if pilot:
        man = harness.build_manifest(OPEN_MODELS[:1] + FRONTIER_MODELS[:1],
                                     conditions, item_ids[:2], [0], 2)
    else:
        man = (harness.build_manifest(OPEN_MODELS, conditions, item_ids,
                                      OPEN_FRAMINGS, OPEN_REPS)
               + harness.build_manifest(FRONTIER_MODELS, conditions, item_ids,
                                        FRONTIER_FRAMINGS, FRONTIER_REPS))
    random.Random(20260815).shuffle(man)  # interleave conditions/models/arms
    return man


def _status_path_counts(manifest: list[dict]) -> dict:
    remaining = [s for s in manifest
                 if not os.path.exists(os.path.join(harness.RAW_DIR, f"{s['run_id']}.json"))]
    return {"total": len(manifest), "completed": len(manifest) - len(remaining),
            "remaining": len(remaining)}


def _write_status(manifest: list[dict]) -> None:
    st = {**_status_path_counts(manifest), "failed_this_session": _failed,
          "updated": time.strftime("%Y-%m-%d %H:%M:%S")}
    os.makedirs(runlog.LOG_DIR, exist_ok=True)
    with open(os.path.join(runlog.LOG_DIR, "manifest-status.json"), "w") as f:
        json.dump(st, f, indent=1)


def _worker(spec: dict, manifest: list[dict]) -> None:
    global _done, _failed
    try:
        harness.run_one(spec)
        with _lock:
            _done += 1
    except Exception:
        with _lock:
            _failed += 1  # already logged by harness; grid keeps going
    with _lock:
        if (_done + _failed) % 25 == 0:
            _write_status(manifest)
            print(f"progress: {_done} done, {_failed} failed this session")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--include-tier2", action="store_true")
    args = ap.parse_args()

    conditions = CONDITIONS_T2 if args.include_tier2 else CONDITIONS_T1
    manifest = build_full_manifest(conditions, pilot=args.pilot)
    counts = _status_path_counts(manifest)
    print(f"manifest: {counts['total']} runs "
          f"({counts['completed']} already done, {counts['remaining']} to go)")

    if args.dry_run:
        by_model: dict[str, int] = {}
        for s in manifest:
            by_model[s["model"]] = by_model.get(s["model"], 0) + 1
        for m, n in sorted(by_model.items()):
            pin, pout = runlog.PRICES.get(m, (0, 0))
            est = n * (11000 / 1e6 * pin + 1500 / 1e6 * pout)  # rough per-run tokens
            print(f"  {m:45} {n:5} runs  ~${est:.2f}")
        return

    runlog.snapshot_config("grid", {
        "conditions": conditions, "open_models": OPEN_MODELS,
        "frontier_models": FRONTIER_MODELS, "open_framings": OPEN_FRAMINGS,
        "open_reps": OPEN_REPS, "frontier_framings": FRONTIER_FRAMINGS,
        "frontier_reps": FRONTIER_REPS, "pilot": args.pilot,
        "temperature": harness.TEMPERATURE,
        "config_version": harness.CONFIG_VERSION,
        "n_items": len(ITEMS)})
    runlog.log_event("grid_start", **counts, pilot=args.pilot)

    by_provider: dict[str, list[dict]] = {}
    for s in manifest:
        by_provider.setdefault(s["provider"], []).append(s)

    threads = []
    for provider, specs in by_provider.items():
        def run_provider(p: str = provider, ss: list[dict] = specs) -> None:
            with cf.ThreadPoolExecutor(max_workers=MAX_CONCURRENCY[p]) as ex:
                list(ex.map(lambda s: _worker(s, manifest), ss))
        t = threading.Thread(target=run_provider, daemon=False)
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

    _write_status(manifest)
    runlog.log_event("grid_end", done=_done, failed=_failed)
    print(f"grid finished: {_done} done, {_failed} failed this session. "
          f"Status: logs/manifest-status.json; costs: logs/costs.json")


if __name__ == "__main__":
    main()
