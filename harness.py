"""Act-then-report harness (v2). Raw generations persisted separately from any
scoring — this module never scores.

Run record schema (runs/raw/<run_id>.json) — the contract for scoring.py:
{
  "run_id": str,              # sha1 of the spec + CONFIG_VERSION
  "arm": "revealed" | "stated",
  "condition": "A"|"B"|"C"|"D",
  "model": str, "provider": "together"|"anthropic"|"openrouter",
  "item_id": str, "framing_idx": 0|1|2,
  "option_order": "orig"|"swapped",   # swapped => options[1] displayed as Option 1
  "displayed": {"1": content_key, "2": content_key},
  "rep": int, "temperature": float,
  "turns": [ {"purpose": "action"|"consequence"|"report"|"stated"|"identity",
              "result": <providers.py call dict, incl. raw response>} ],
  "logged_action": {"present": bool, "option_position": "1"|"2"|None,
                    "content_key": str|None, "n_tool_calls": int},
  "config_version": str,
}
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
from typing import Any

import providers
import runlog
from conditions import (CONDITIONS, DOCUMENT_FRAME_TEMPLATE, IDENTITY_PROBES,
                        REPORT_QUESTION)
from items_draft import ITEMS, RECORD_CHOICE_TOOL

CONFIG_VERSION = "v2-draft-2026-08-14"
TEMPERATURE = 0.7
RAW_DIR = "runs/raw"


def _call(provider: str, model: str, messages: list[dict],
          tools: list[dict] | None = None, max_tokens: int = 700) -> dict:
    if provider == "together":
        result = providers.call_together(model, messages, tools=tools,
                                         temperature=TEMPERATURE, max_tokens=max_tokens)
    elif provider == "anthropic":
        result = providers.call_anthropic(model, messages, tools=tools,
                                          temperature=TEMPERATURE, max_tokens=max_tokens)
    elif provider == "openrouter":
        result = providers.call_openrouter(model, messages, tools=tools,
                                           temperature=TEMPERATURE, max_tokens=max_tokens)
    else:
        raise ValueError(provider)
    runlog.add_cost(model, result)
    if not result["ok"]:
        runlog.log_event("api_error", provider=provider, model=model,
                         status=result["status"],
                         error=str((result.get("raw") or {}).get("error", ""))[:200])
    return result


def _item(item_id: str) -> dict:
    return next(i for i in ITEMS if i["id"] == item_id)


def _displayed(item: dict, option_order: str) -> dict[str, dict]:
    a, b = item["options"]
    return {"1": a, "2": b} if option_order == "orig" else {"1": b, "2": a}


def _option_block(disp: dict[str, dict]) -> str:
    return (f"Option 1: {disp['1']['text']}\n"
            f"Option 2: {disp['2']['text']}")


def _action_user_turn(spec: dict, item: dict, disp: dict[str, dict]) -> str:
    framing = item["framings"][spec["framing_idx"]]
    task_block = framing.format(options=_option_block(disp))
    cond = CONDITIONS[spec["condition"]]
    parts = []
    if cond["prefix_instruction"]:
        parts.append(cond["prefix_instruction"])
    if cond["document_frame"]:
        return DOCUMENT_FRAME_TEMPLATE.format(task_block=task_block)
    parts.append(task_block)
    return "\n\n".join(parts)


def run_id_for(spec: dict) -> str:
    key = json.dumps({**spec, "config": CONFIG_VERSION}, sort_keys=True)
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def build_manifest(models: list[dict], conditions: list[str], item_ids: list[str],
                   framing_idxs: list[int], n_reps: int,
                   arms: tuple[str, ...] = ("revealed", "stated")) -> list[dict]:
    """models: [{"model": id, "provider": name}]. Option order alternates by rep
    (odd rep = orig, even rep = swapped) — counterbalanced within every cell."""
    manifest = []
    for arm in arms:
        for m in models:
            for cond in conditions:
                for item_id in item_ids:
                    for f_idx in framing_idxs:
                        for rep in range(1, n_reps + 1):
                            spec = {
                                "arm": arm, "condition": cond,
                                "model": m["model"], "provider": m["provider"],
                                "item_id": item_id, "framing_idx": f_idx,
                                "option_order": "orig" if rep % 2 == 1 else "swapped",
                                "rep": rep,
                            }
                            spec["run_id"] = run_id_for(spec)
                            manifest.append(spec)
    # Interleave execution across conditions/models so provider drift is not
    # confounded with condition (deterministic shuffle, seeded).
    random.Random(20260814).shuffle(manifest)
    return manifest


def _extract_action(result: dict, disp: dict[str, dict]) -> dict:
    calls = [c for c in result.get("tool_calls", []) if c.get("name") == "record_choice"]
    first_valid = next((c for c in calls
                        if str(c.get("arguments", {}).get("option")) in ("1", "2")), None)
    pos = str(first_valid["arguments"]["option"]) if first_valid else None
    return {"present": first_valid is not None,
            "option_position": pos,
            "content_key": disp[pos]["key"] if pos else None,
            "n_tool_calls": len(result.get("tool_calls", []))}


def _report_labels(spec: dict) -> tuple[str, str]:
    # Counterbalance the report answer order against a fixed rule tied to rep,
    # independent of the displayed option order.
    return (("Option 1", "Option 2") if spec["rep"] % 4 in (1, 2)
            else ("Option 2", "Option 1"))


def run_revealed(spec: dict, identity_probe: bool = False) -> dict:
    item = _item(spec["item_id"])
    disp = _displayed(item, spec["option_order"])
    cond = CONDITIONS[spec["condition"]]
    turns: list[dict] = []

    messages: list[dict] = []
    if cond["system"]:
        messages.append({"role": "system", "content": cond["system"]})
    messages.extend(cond.get("primed_history", []))  # Condition E only
    messages.append({"role": "user", "content": _action_user_turn(spec, item, disp)})

    # Action turn — tools available, NOT forced (forcing only in some conditions
    # would be an instrument confound; B needs prefix text + tool in one turn).
    action = _call(spec["provider"], spec["model"], messages, tools=[RECORD_CHOICE_TOOL])
    turns.append({"purpose": "action", "result": action})
    logged = _extract_action(action, disp)

    if logged["present"]:
        chosen = disp[logged["option_position"]]
        # Neutral ack — NEVER echoes the chosen option (design rule).
        messages.append({"role": "assistant", "content": action["text"] or "[tool call]"})
        messages.append({"role": "user",
                         "content": f"Recorded. {chosen['consequence']}"})
        consequence = _call(spec["provider"], spec["model"], messages, max_tokens=400)
        turns.append({"purpose": "consequence", "result": consequence})
        messages.append({"role": "assistant",
                         "content": consequence["text"] or "[no text]"})

        first, second = _report_labels(spec)
        messages.append({"role": "user",
                         "content": REPORT_QUESTION.format(first=first, second=second)})
        report = _call(spec["provider"], spec["model"], messages, max_tokens=50)
        turns.append({"purpose": "report", "result": report})

        if identity_probe:
            messages.append({"role": "assistant", "content": report["text"] or "[no text]"})
            for probe in IDENTITY_PROBES:
                messages.append({"role": "user", "content": probe})
                pr = _call(spec["provider"], spec["model"], messages, max_tokens=120)
                turns.append({"purpose": "identity", "result": pr})
                messages.append({"role": "assistant", "content": pr["text"] or "[no text]"})

    record = {**spec, "displayed": {k: v["key"] for k, v in disp.items()},
              "temperature": TEMPERATURE, "turns": turns,
              "logged_action": logged, "config_version": CONFIG_VERSION}
    providers.persist(os.path.join(RAW_DIR, f"{spec['run_id']}.json"), record)
    return record


def run_stated(spec: dict) -> dict:
    """Fresh-context stated preference: same condition preamble, same item,
    one turn, no tools, abstention allowed."""
    from conditions import STATED_QUESTION
    item = _item(spec["item_id"])
    disp = _displayed(item, spec["option_order"])
    cond = CONDITIONS[spec["condition"]]

    first, second = "Option 1", "Option 2"
    q = STATED_QUESTION.format(option_block=_option_block(disp),
                               first=first, second=second)
    if cond["prefix_instruction"]:
        q = cond["prefix_instruction"] + "\n\n" + q
    if cond["document_frame"]:
        q = DOCUMENT_FRAME_TEMPLATE.format(task_block=q)

    messages: list[dict] = []
    if cond["system"]:
        messages.append({"role": "system", "content": cond["system"]})
    messages.extend(cond.get("primed_history", []))  # Condition E only
    messages.append({"role": "user", "content": q})
    result = _call(spec["provider"], spec["model"], messages, max_tokens=30)

    record = {**spec, "displayed": {k: v["key"] for k, v in disp.items()},
              "temperature": TEMPERATURE,
              "turns": [{"purpose": "stated", "result": result}],
              "logged_action": {"present": False, "option_position": None,
                                "content_key": None, "n_tool_calls": 0},
              "config_version": CONFIG_VERSION}
    providers.persist(os.path.join(RAW_DIR, f"{spec['run_id']}.json"), record)
    return record


def run_one(spec: dict) -> dict:
    path = os.path.join(RAW_DIR, f"{spec['run_id']}.json")
    if os.path.exists(path):  # idempotent — safe to resume a killed grid
        with open(path) as f:
            return json.load(f)
    t0 = time.time()
    try:
        if spec["arm"] == "stated":
            record = run_stated(spec)
        else:
            record = run_revealed(spec, identity_probe=(spec["rep"] == 1))
    except Exception as e:
        runlog.log_event("run_failed", run_id=spec["run_id"], model=spec["model"],
                         condition=spec["condition"], arm=spec["arm"], error=repr(e)[:300])
        raise
    runlog.log_event("run_done", run_id=spec["run_id"], model=spec["model"],
                     condition=spec["condition"], arm=spec["arm"],
                     item=spec["item_id"], rep=spec["rep"],
                     action=record["logged_action"].get("content_key"),
                     n_turns=len(record["turns"]),
                     secs=round(time.time() - t0, 2))
    return record
