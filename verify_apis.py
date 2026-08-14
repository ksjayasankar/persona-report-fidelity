#!/usr/bin/env python3
"""Live, non-destructive capability checks for the three experiment APIs."""

from __future__ import annotations

import argparse
import os
import re
from typing import Any, Callable

import requests

import providers
from conditions import PREFIX_TEXT

OUT_DIR = "runs/verification"
RESULTS: list[str] = []
TOOL = {
    "type": "function",
    "function": {
        "name": "record_choice",
        "description": "Record one of the two options.",
        "parameters": {
            "type": "object",
            "properties": {"option": {"type": "string", "enum": ["1", "2"]}},
            "required": ["option"],
            "additionalProperties": False,
        },
    },
}
QUESTION = "Choose one: 1) red, or 2) blue. Record the choice with record_choice."
FORCED = {"type": "function", "function": {"name": "record_choice"}}

def emit(kind: str, name: str, detail: str, record: Any | None = None) -> None:
    line = f"{kind} {name}: {detail}"
    print(line)
    RESULTS.append(line)
    if record is not None:
        providers.persist(f"{OUT_DIR}/{name}.json", record)

def skip(provider: str) -> None:
    line = f"SKIP {provider}"
    print(line)
    RESULTS.append(line)

def get_json(url: str, headers: dict[str, str]) -> tuple[int, Any]:
    try:
        response = requests.get(url, headers=headers, timeout=60)
        try:
            return response.status_code, response.json()
        except ValueError:
            return response.status_code, {"_non_json_body": response.text[:2000]}
    except requests.RequestException as exc:
        return -1, {"_transport_error": repr(exc)}

def model_ids(body: Any) -> list[str]:
    rows = body if isinstance(body, list) else body.get("data", body.get("models", []))
    return sorted({str(row["id"]) for row in rows if isinstance(row, dict) and row.get("id")})

def valid_choice(result: dict[str, Any]) -> bool:
    return any(
        call.get("name") == "record_choice"
        and str(call.get("arguments", {}).get("option")) in {"1", "2"}
        for call in result.get("tool_calls", [])
    )

def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")

def compact(text: str, limit: int = 70) -> str:
    return re.sub(r"\s+", " ", text).strip()[:limit] or "<empty>"

def together_checks(key: str) -> None:
    status, body = get_json(
        f"{providers.TOGETHER_BASE}/models", {"Authorization": f"Bearer {key}"}
    )
    ids = model_ids(body) if status == 200 else []
    low = {model: model.lower() for model in ids}
    rules: dict[str, Callable[[str], bool]] = {
        "gpt-oss-120b": lambda s: "gpt-oss-120b" in s,
        "gpt-oss-20b": lambda s: "gpt-oss-20b" in s,
        "qwen3-235b-2507": lambda s: "qwen3-235b-a22b-instruct" in s and "2507" in s,
        "llama-4-scout": lambda s: "llama" in s and "4" in s and "scout" in s,
        "minimax-m2": lambda s: "minimax" in s and re.search(r"m2(?:[._-]|$)", s) is not None,
        "deepseek": lambda s: "deepseek" in s,
        "kimi": lambda s: "kimi" in s,
        "small-qwen-instruct": lambda s: "instruct" in s
        and (size := re.search(r"qwen[\d.]*[-_/](\d+(?:\.\d+)?)b", s)) is not None
        and float(size.group(1)) <= 32,
    }
    found = {label: [model for model, lowered in low.items() if rule(lowered)]
             for label, rule in rules.items()}
    detail = "; ".join(f"{label}={','.join(matches) or 'MISSING'}"
                       for label, matches in found.items())
    emit("PASS" if status == 200 else "FAIL", "together-models",
         detail if status == 200 else f"HTTP {status}",
         {"status": status, "raw": body, "matches": found})

    panel = list(dict.fromkeys(model for label in (
        "gpt-oss-120b", "gpt-oss-20b", "qwen3-235b-2507", "llama-4-scout", "minimax-m2"
    ) for model in found[label]))
    small = (found["gpt-oss-20b"] + found["small-qwen-instruct"] + panel)
    if small:
        result = providers.call_together(
            small[0], [{"role": "user", "content": "Complete the fact."},
                       {"role": "assistant", "content": "The capital of France is"}],
            temperature=0, max_tokens=16,
        )
        if not result["ok"]:
            outcome = f"errors (HTTP {result['status']})"
        elif result["text"].lstrip().lower().startswith("paris"):
            outcome = f"continues turn ({compact(result['text'])})"
        else:
            outcome = f"starts fresh ({compact(result['text'])})"
        emit("INFO", "together-continuation", outcome, result)
    else:
        emit("INFO", "together-continuation", "errors (no live small model)",
             {"status": None, "raw": None})

    for model in panel:
        result = providers.call_together(
            model, [{"role": "user", "content": QUESTION}], tools=[TOOL],
            tool_choice=FORCED, temperature=0, max_tokens=64,
        )
        name = f"together-toolcall-{slug(model)}"
        ok = result["ok"] and valid_choice(result)
        emit("PASS" if ok else "FAIL", name,
             f"model={model}; HTTP {result['status']}; valid_choice={valid_choice(result)}", result)

    prefix_model = next((found[name][0] for name in (
        "gpt-oss-120b", "qwen3-235b-2507", "llama-4-scout", "minimax-m2", "gpt-oss-20b"
    ) if found[name]), None)
    prefix_plus_tool("together", prefix_model, providers.call_together)

    log_model = (found["small-qwen-instruct"] + found["gpt-oss-20b"] + panel)
    if not log_model:
        emit("INFO", "together-logprobs", "no live model", {"raw": None})
        return
    chat = providers.call_together(
        log_model[0], [{"role": "user", "content": "Reply with one word: Paris"}],
        temperature=0, max_tokens=8, extra={"logprobs": 1},
    )
    completion = providers.call_together_completions(
        log_model[0], "The capital of France is", temperature=0, max_tokens=8,
        logprobs=1, echo=True,
    )
    chat_lp = ((chat.get("raw", {}).get("choices") or [{}])[0].get("logprobs"))
    comp_lp = ((completion.get("raw", {}).get("choices") or [{}])[0].get("logprobs"))
    chat_keys = sorted(chat_lp) if isinstance(chat_lp, dict) else []
    comp_keys = sorted(comp_lp) if isinstance(comp_lp, dict) else []
    emit("INFO", "together-logprobs", f"chat keys={chat_keys}; completion keys={comp_keys}",
         {"chat": chat, "completions": completion})

def prefix_plus_tool(provider: str, model: str | None,
                     caller: Callable[..., dict[str, Any]]) -> None:
    name = f"prefix-plus-tool-{provider}"
    if not model:
        emit("INFO", name, "no live model", {"raw": None})
        return
    prompt = f'Begin your reply with exactly "{PREFIX_TEXT}" Then {QUESTION}'
    result = caller(model, [{"role": "user", "content": prompt}], tools=[TOOL],
                    temperature=0, max_tokens=96)
    has_prefix = PREFIX_TEXT in result.get("text", "")
    has_tool = valid_choice(result)
    emit("INFO", name,
         f"model={model}; HTTP {result['status']}; prefix={has_prefix}; valid_tool={has_tool}", result)

def anthropic_checks(key: str) -> None:
    status, body = get_json(
        f"{providers.ANTHROPIC_BASE}/models",
        {"x-api-key": key, "anthropic-version": "2023-06-01"},
    )
    ids = model_ids(body) if status == 200 else []
    matches = {needle: [model for model in ids if needle in model.lower()]
               for needle in ("haiku-4-5", "sonnet-5", "sonnet-4-5")}
    detail = "; ".join(f"{name}={','.join(values) or 'MISSING'}"
                       for name, values in matches.items())
    emit("PASS" if status == 200 else "FAIL", "anthropic-models",
         detail if status == 200 else f"HTTP {status}",
         {"status": status, "raw": body, "matches": matches})
    haiku = matches["haiku-4-5"][0] if matches["haiku-4-5"] else None
    prefix_plus_tool("anthropic", haiku, providers.call_anthropic)
    anthropic_prefill("anthropic-prefill-haiku", haiku, expect_400=False)
    anthropic_prefill("anthropic-prefill-haiku-whitespace", haiku, expect_400=True,
                      trailing="The capital of France is ")
    sonnet = matches["sonnet-5"][0] if matches["sonnet-5"] else None
    anthropic_prefill("anthropic-prefill-sonnet5", sonnet, expect_400=True)

def anthropic_prefill(name: str, model: str | None, expect_400: bool,
                      trailing: str = "The capital of France is") -> None:
    if not model:
        emit("INFO", name, "no live matching model", {"raw": None})
        return
    result = providers.call_anthropic(
        model, [{"role": "user", "content": "Complete the fact."},
                {"role": "assistant", "content": trailing}],
        temperature=0, max_tokens=16,
    )
    if expect_400:
        ok = result["status"] == 400
        detail = f"model={model}; HTTP {result['status']}; expected 400"
    else:
        ok = result["ok"] and result["text"].lstrip().lower().startswith("paris")
        detail = f"model={model}; HTTP {result['status']}; continuation={compact(result['text'])}"
    emit("PASS" if ok else "FAIL", name, detail, result)

def openrouter_checks() -> None:
    model = "openai/gpt-5.1"
    prefix_plus_tool("openrouter", model, providers.call_openrouter)
    basic = providers.call_openrouter(
        model, [{"role": "user", "content": "Reply with exactly OK."}],
        temperature=0, max_tokens=16,
    )
    tool = providers.call_openrouter(
        model, [{"role": "user", "content": QUESTION}], tools=[TOOL],
        tool_choice=FORCED, temperature=0, max_tokens=64,
    )
    def upstream(result: dict[str, Any]) -> str:
        raw = result.get("raw", {})
        return str(raw.get("provider") or raw.get("metadata", {}).get("provider") or "unknown")
    ok = basic["ok"] and tool["ok"] and valid_choice(tool)
    emit("PASS" if ok else "FAIL", "openrouter-gpt51",
         f"basic HTTP {basic['status']} via {upstream(basic)}; tool HTTP {tool['status']} "
         f"via {upstream(tool)}; valid_choice={valid_choice(tool)}",
         {"basic": basic, "forced_tool": tool})

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=("together", "anthropic", "openrouter"))
    args = parser.parse_args()
    checks = {
        "together": ("TOGETHER_API_KEY", together_checks),
        "anthropic": ("ANTHROPIC_API_KEY", anthropic_checks),
        "openrouter": ("OPENROUTER_API_KEY", lambda _key: openrouter_checks()),
    }
    for provider, (env_name, check) in checks.items():
        if args.only and args.only != provider:
            continue
        key = os.environ.get(env_name)
        if not key:
            skip(provider)
            continue
        check(key)
    print("\nSUMMARY")
    for line in RESULTS:
        print(line)


if __name__ == "__main__":
    main()
