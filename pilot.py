"""Pilot: 2 models x 2 items x 4 conditions x 2 reps, both arms (~64 runs, <$1).

Run AFTER verify_apis.py passes. Pilot data never enters confirmatory analysis
(pre-registered). Usage: python3 pilot.py [--models together|all]
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import harness

PILOT_MODELS = {
    "together": [{"model": "openai/gpt-oss-120b", "provider": "together"}],
    "all": [
        # Open panel (verified accessible + tool-calling 2026-08-14; Llama and
        # Qwen3-235B are dedicated-only on this account and unavailable).
        {"model": "openai/gpt-oss-120b", "provider": "together"},
        {"model": "MiniMaxAI/MiniMax-M3", "provider": "together"},
        {"model": "deepseek-ai/DeepSeek-V4-Flash-0731", "provider": "together"},
        {"model": "Qwen/Qwen3.7-Plus", "provider": "together"},  # stream-only
        {"model": "claude-haiku-4-5-20251001", "provider": "anthropic"},
        # Exploratory candidate (Tier 2, NOT confirmatory panel). Gate on three
        # pilot checks: (1) temperature 0.7 must not 400 or silently no-op
        # (Gemini 3.6+ deprecated sampling params); (2) the flattened tool loop
        # completes without thought_signature errors; (3) valid record_choice.
        {"model": "google/gemini-3.7-flash", "provider": "openrouter"},
    ],
}
PILOT_ITEMS = ["tedium_vs_creative", "donation_nets_vs_animals"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", choices=list(PILOT_MODELS), default="all")
    args = ap.parse_args()

    manifest = harness.build_manifest(
        PILOT_MODELS[args.models], ["A", "B", "C", "D", "E"], PILOT_ITEMS,
        framing_idxs=[0], n_reps=2)
    print(f"pilot: {len(manifest)} runs")

    ok = fail = 0
    for i, spec in enumerate(manifest, 1):
        try:
            rec = harness.run_one(spec)
            action = rec["logged_action"]
            status = ("action=" + str(action["content_key"])
                      if action["present"] else "NO-ACTION")
            if spec["arm"] == "stated":
                status = "stated:" + (rec["turns"][0]["result"]["text"] or "")[:40].replace("\n", " ")
            ok += 1
            print(f"[{i}/{len(manifest)}] {spec['model'].split('/')[-1]} "
                  f"{spec['condition']} {spec['item_id']} {spec['arm']} -> {status}")
        except Exception as e:  # keep going; pilot exists to surface failures
            fail += 1
            print(f"[{i}/{len(manifest)}] FAILED {spec['run_id']}: {e!r}",
                  file=sys.stderr)
        time.sleep(0.3)

    print(f"\npilot done: {ok} ok, {fail} failed. Raw records in runs/raw/. "
          f"Inspect NO-ACTION rates per condition before locking anything.")
    summary = {"ok": ok, "failed": fail, "n": len(manifest)}
    with open("runs/pilot_summary.json", "w") as f:
        json.dump(summary, f)


if __name__ == "__main__":
    main()
