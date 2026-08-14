"""Thin provider clients: Together / Anthropic / OpenRouter.

All calls return a dict: {"ok", "status", "raw" (full response json), "text",
"tool_calls" (normalized [{name, arguments}]), "request" (what we sent),
"meta" (model, provider, params, timestamps, attempt)}.
Raw responses are persisted by the caller — this module never scores anything.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import requests

TOGETHER_BASE = "https://api.together.xyz/v1"
ANTHROPIC_BASE = "https://api.anthropic.com/v1"
OPENROUTER_BASE = "https://openrouter.ai/api/v1"

RETRYABLE = {429, 500, 502, 503, 529}
MAX_ATTEMPTS = 5


def _key(name: str) -> str:
    v = os.environ.get(name, "")
    if not v:
        raise RuntimeError(f"missing env var {name}")
    return v


def _post(url: str, headers: dict[str, str], payload: dict[str, Any],
          timeout: int = 180) -> tuple[int, dict[str, Any]]:
    last_status, last_body = 0, {}
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=timeout)
            last_status = r.status_code
            try:
                last_body = r.json()
            except ValueError:
                last_body = {"_non_json_body": r.text[:2000]}
            if r.status_code not in RETRYABLE:
                return last_status, last_body
        except requests.RequestException as e:
            last_status, last_body = -1, {"_transport_error": repr(e)}
        time.sleep(min(2 ** attempt, 30))
    return last_status, last_body


def _openai_style(base: str, key_env: str, model: str, messages: list[dict],
                  tools: list[dict] | None, tool_choice: Any,
                  temperature: float, max_tokens: int,
                  extra: dict[str, Any] | None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if tools:
        payload["tools"] = tools
    if tool_choice is not None:
        payload["tool_choice"] = tool_choice
    if extra:
        payload.update(extra)
    t0 = time.time()
    status, body = _post(f"{base}/chat/completions",
                         {"Authorization": f"Bearer {_key(key_env)}"}, payload)
    text, tool_calls = "", []
    if status == 200:
        msg = (body.get("choices") or [{}])[0].get("message", {})
        text = msg.get("content") or ""
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function", {})
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except ValueError:
                args = {"_unparseable": fn.get("arguments")}
            tool_calls.append({"name": fn.get("name"), "arguments": args})
    return {"ok": status == 200, "status": status, "raw": body, "text": text,
            "tool_calls": tool_calls, "request": payload,
            "meta": {"provider": base, "model": model, "t_start": t0,
                     "t_end": time.time(), "temperature": temperature}}


# Models that reject non-streaming requests ("This model only supports streaming").
TOGETHER_STREAM_ONLY = {"Qwen/Qwen3.7-Plus"}


def _together_stream(model: str, payload: dict[str, Any]) -> dict[str, Any]:
    """SSE call; reassembles text + tool_calls into the normal result shape."""
    payload = {**payload, "stream": True}
    t0 = time.time()
    text, calls, last_status = "", {}, -1
    raw_meta: dict[str, Any] = {}
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            r = requests.post(f"{TOGETHER_BASE}/chat/completions",
                              headers={"Authorization": f"Bearer {_key('TOGETHER_API_KEY')}"},
                              json=payload, stream=True, timeout=180)
            last_status = r.status_code
            if r.status_code in RETRYABLE:
                time.sleep(min(2 ** attempt, 30))
                continue
            if r.status_code != 200:
                try:
                    return {"ok": False, "status": r.status_code, "raw": r.json(),
                            "text": "", "tool_calls": [], "request": payload,
                            "meta": {"provider": "together/stream", "model": model,
                                     "t_start": t0, "t_end": time.time()}}
                except ValueError:
                    break
            text, calls = "", {}
            for line in r.iter_lines():
                if not line or not line.startswith(b"data: "):
                    continue
                if line[6:] == b"[DONE]":
                    break
                try:
                    chunk = json.loads(line[6:])
                except ValueError:
                    continue
                raw_meta = {k: chunk.get(k) for k in ("id", "model", "usage") if chunk.get(k)}
                delta = (chunk.get("choices") or [{}])[0].get("delta", {})
                text += delta.get("content") or ""
                for tc in delta.get("tool_calls") or []:
                    idx = tc.get("index", 0)
                    slot = calls.setdefault(idx, {"name": "", "args": ""})
                    fn = tc.get("function", {})
                    slot["name"] = fn.get("name") or slot["name"]
                    slot["args"] += fn.get("arguments") or ""
            tool_calls = []
            for slot in calls.values():
                try:
                    args = json.loads(slot["args"] or "{}")
                except ValueError:
                    args = {"_unparseable": slot["args"]}
                tool_calls.append({"name": slot["name"], "arguments": args})
            return {"ok": True, "status": 200,
                    "raw": {"_reassembled_stream": True, **raw_meta,
                            "text": text, "tool_calls": tool_calls},
                    "text": text, "tool_calls": tool_calls, "request": payload,
                    "meta": {"provider": "together/stream", "model": model,
                             "t_start": t0, "t_end": time.time(),
                             "temperature": payload.get("temperature")}}
        except requests.RequestException as e:
            last_status = -1
            time.sleep(min(2 ** attempt, 30))
    return {"ok": False, "status": last_status, "raw": {"_stream_failed": True},
            "text": "", "tool_calls": [], "request": payload,
            "meta": {"provider": "together/stream", "model": model,
                     "t_start": t0, "t_end": time.time()}}


def call_together(model: str, messages: list[dict], tools: list[dict] | None = None,
                  tool_choice: Any = None, temperature: float = 0.7,
                  max_tokens: int = 1024, extra: dict | None = None) -> dict:
    if model in TOGETHER_STREAM_ONLY:
        payload: dict[str, Any] = {"model": model, "messages": messages,
                                   "temperature": temperature, "max_tokens": max_tokens}
        if tools:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        if extra:
            payload.update(extra)
        return _together_stream(model, payload)
    return _openai_style(TOGETHER_BASE, "TOGETHER_API_KEY", model, messages,
                         tools, tool_choice, temperature, max_tokens, extra)


def call_openrouter(model: str, messages: list[dict], tools: list[dict] | None = None,
                    tool_choice: Any = None, temperature: float = 0.7,
                    max_tokens: int = 1024, extra: dict | None = None) -> dict:
    # Pin provider routing where possible (set in `extra`, e.g. {"provider": {"order": ["openai"]}}).
    return _openai_style(OPENROUTER_BASE, "OPENROUTER_API_KEY", model, messages,
                         tools, tool_choice, temperature, max_tokens, extra)


def call_together_completions(model: str, prompt: str, temperature: float = 0.7,
                              max_tokens: int = 1024, logprobs: int | None = None,
                              echo: bool = False) -> dict:
    payload: dict[str, Any] = {"model": model, "prompt": prompt,
                               "temperature": temperature, "max_tokens": max_tokens}
    if logprobs is not None:
        payload["logprobs"] = logprobs
    if echo:
        payload["echo"] = True
    t0 = time.time()
    status, body = _post(f"{TOGETHER_BASE}/completions",
                         {"Authorization": f"Bearer {_key('TOGETHER_API_KEY')}"}, payload)
    text = ""
    if status == 200:
        text = (body.get("choices") or [{}])[0].get("text") or ""
    return {"ok": status == 200, "status": status, "raw": body, "text": text,
            "tool_calls": [], "request": payload,
            "meta": {"provider": "together/completions", "model": model,
                     "t_start": t0, "t_end": time.time(), "temperature": temperature}}


def call_anthropic(model: str, messages: list[dict], system: str | None = None,
                   tools: list[dict] | None = None, tool_choice: Any = None,
                   temperature: float = 0.7, max_tokens: int = 1024) -> dict:
    """messages use OpenAI shape; converted here. A trailing assistant message
    becomes a prefill (legal only on prefill-capable models)."""
    conv = [{"role": m["role"], "content": m["content"]}
            for m in messages if m["role"] != "system"]
    payload: dict[str, Any] = {"model": model, "messages": conv,
                               "temperature": temperature, "max_tokens": max_tokens}
    sys_msgs = [m["content"] for m in messages if m["role"] == "system"]
    if system or sys_msgs:
        payload["system"] = system or "\n\n".join(sys_msgs)
    if tools:
        payload["tools"] = [{"name": t["function"]["name"],
                             "description": t["function"].get("description", ""),
                             "input_schema": t["function"]["parameters"]}
                            for t in tools]
    if tool_choice is not None:
        payload["tool_choice"] = tool_choice  # e.g. {"type": "tool", "name": "record_choice"}
    t0 = time.time()
    status, body = _post(f"{ANTHROPIC_BASE}/messages",
                         {"x-api-key": _key("ANTHROPIC_API_KEY"),
                          "anthropic-version": "2023-06-01"}, payload)
    text, tool_calls = "", []
    if status == 200:
        for block in body.get("content", []):
            if block.get("type") == "text":
                text += block.get("text", "")
            elif block.get("type") == "tool_use":
                tool_calls.append({"name": block.get("name"),
                                   "arguments": block.get("input", {})})
    return {"ok": status == 200, "status": status, "raw": body, "text": text,
            "tool_calls": tool_calls, "request": payload,
            "meta": {"provider": "anthropic", "model": model, "t_start": t0,
                     "t_end": time.time(), "temperature": temperature}}


def persist(path: str, record: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(record, f, indent=1, default=str)
