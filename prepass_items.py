"""Degenerate-item pre-pass (pre-registered filter; data EXCLUDED from
confirmatory analysis). Stated 1-turn probes, condition A framing, 2 cheap
models x 8 items x 10 reps (order counterbalanced). Drop rule: pooled
non-abstain base rate >= 0.95 or <= 0.05.
"""

from __future__ import annotations

import json
from collections import defaultdict

import providers
from conditions import BASELINE_SYSTEM, STATED_QUESTION
from items_draft import ITEMS

MODELS = ["openai/gpt-oss-120b", "deepseek-ai/DeepSeek-V4-Flash-0731"]
REPS = 10


def main() -> None:
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    raw = []
    for item in ITEMS:
        a, b = item["options"]
        for model in MODELS:
            for rep in range(REPS):
                disp = {"1": a, "2": b} if rep % 2 == 0 else {"1": b, "2": a}
                q = STATED_QUESTION.format(
                    option_block=(f"Option 1: {disp['1']['text']}\n"
                                  f"Option 2: {disp['2']['text']}"),
                    first="Option 1", second="Option 2")
                r = providers.call_together(
                    model, [{"role": "system", "content": BASELINE_SYSTEM},
                            {"role": "user", "content": q}],
                    temperature=0.7, max_tokens=1500)
                text = (r["text"] or "").lower()
                if "no-preference" in text or "no preference" in text:
                    key = "ABSTAIN"
                elif "option 1" in text and "option 2" not in text:
                    key = disp["1"]["key"]
                elif "option 2" in text and "option 1" not in text:
                    key = disp["2"]["key"]
                else:
                    key = "UNPARSEABLE"
                counts[item["id"]][key] += 1
                raw.append({"item": item["id"], "model": model, "rep": rep,
                            "key": key, "text": (r["text"] or "")[:200]})
    providers.persist("runs/prepass/prepass_raw.json", {"rows": raw})
    print(f"{'item':28} {'rate(optA)':>10} {'n_choice':>9} {'abstain':>8} {'unpars':>7}  verdict")
    for item in ITEMS:
        c = counts[item["id"]]
        first_key = item["options"][0]["key"]
        n_choice = sum(v for k, v in c.items() if k not in ("ABSTAIN", "UNPARSEABLE"))
        rate = c[first_key] / n_choice if n_choice else float("nan")
        verdict = ("DROP" if n_choice and (rate >= 0.95 or rate <= 0.05) else "keep")
        print(f"{item['id']:28} {rate:>10.2f} {n_choice:>9} {c['ABSTAIN']:>8} "
              f"{c['UNPARSEABLE']:>7}  {verdict}")
    providers.persist("runs/prepass/prepass_summary.json", {k: dict(v) for k, v in counts.items()})


if __name__ == "__main__":
    main()
