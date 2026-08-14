"""Figure builder — regenerates every figure from runs/scored/scores.jsonl.

Nothing hand-drawn, nothing hand-numbered: rerunning this script after a
scoring change rebuilds all PNGs in runs/report/figures/.

  Fig 1  divergence rate per condition x model, Wilson 95% CIs (main figure)
  Fig 2  action-completion (attrition) heatmap, condition x model
  Fig 3  stated vs revealed cross-condition stability (H2), per model
  Fig 4  outcome-state composition per condition (stacked, explicit denominators)
"""

from __future__ import annotations

import json
import math
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCORES = "runs/scored/scores.jsonl"
OUT = "runs/report/figures"
CONDS = ["A", "B", "C", "D", "E"]  # E present only if Tier-2 cells were run


def load() -> list[dict]:
    with open(SCORES) as f:
        return [json.loads(line) for line in f if line.strip()]


def wilson(k: int, n: int) -> tuple[float, float, float]:
    if n == 0:
        return 0.0, 0.0, 0.0
    z, p = 1.959964, k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0.0, centre - half), min(1.0, centre + half)


def short(model: str) -> str:
    return model.split("/")[-1].replace("-Instruct", "").replace("-0731", "")


def fig1_divergence(rows: list[dict]) -> None:
    cells: dict[tuple[str, str], list[int]] = defaultdict(list)
    for r in rows:
        if r.get("arm") != "revealed" or r.get("outcome") not in ("consistent", "divergent"):
            continue
        cells[(r["model"], r["condition"])].append(1 if r["outcome"] == "divergent" else 0)
    models = sorted({m for m, _ in cells})
    conds = [c for c in CONDS if any((m, c) in cells for m in models)]
    fig, ax = plt.subplots(figsize=(1.6 * max(4, len(models) * len(conds) / 3), 4))
    width = 0.8 / max(1, len(conds))
    for ci, cond in enumerate(conds):
        xs, ys, lo, hi = [], [], [], []
        for mi, m in enumerate(models):
            v = cells.get((m, cond), [])
            p, l, h = wilson(sum(v), len(v))
            xs.append(mi + ci * width)
            ys.append(p)
            lo.append(p - l)
            hi.append(h - p)
        ax.bar(xs, ys, width=width, label=f"Condition {cond}")
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt="none", ecolor="black",
                    elinewidth=0.8, capsize=2)
    ax.set_xticks([i + 0.4 - width / 2 for i in range(len(models))])
    ax.set_xticklabels([short(m) for m in models], rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("report/action divergence rate\n(divergent / scored)")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, ncols=len(conds))
    ax.set_title("Fig 1 — Report/action divergence by persona condition (Wilson 95% CI)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig1_divergence.png", dpi=200)
    plt.close(fig)


def fig2_attrition(rows: list[dict]) -> None:
    tot: dict[tuple[str, str], int] = defaultdict(int)
    okc: dict[tuple[str, str], int] = defaultdict(int)
    for r in rows:
        if r.get("arm") != "revealed":
            continue
        key = (r["model"], r["condition"])
        tot[key] += 1
        if r.get("action_valid"):
            okc[key] += 1
    models = sorted({m for m, _ in tot})
    conds = [c for c in CONDS if any((m, c) in tot for m in models)]
    grid = [[(okc[(m, c)] / tot[(m, c)] if tot.get((m, c)) else float("nan"))
             for c in conds] for m in models]
    fig, ax = plt.subplots(figsize=(1.2 * len(conds) + 3, 0.5 * len(models) + 2))
    im = ax.imshow(grid, vmin=0, vmax=1, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(conds)), [f"Cond {c}" for c in conds])
    ax.set_yticks(range(len(models)), [short(m) for m in models], fontsize=8)
    for i, row in enumerate(grid):
        for j, v in enumerate(row):
            if not math.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v < 0.6 else "black", fontsize=8)
    fig.colorbar(im, label="valid-action rate")
    ax.set_title("Fig 2 — Action completion by condition (differential attrition)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig2_attrition.png", dpi=200)
    plt.close(fig)


def _choice_dist(rows: list[dict], arm: str) -> dict[tuple[str, str, str], dict[str, int]]:
    d: dict[tuple[str, str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in rows:
        if r.get("arm") != arm:
            continue
        key_field = "content_key" if arm == "revealed" else "stated_content_key"
        ck = r.get(key_field) or (r.get("logged_action") or {}).get("content_key")
        if ck:
            d[(r["model"], r["item_id"], r["condition"])][ck] += 1
    return d


def fig3_stability(rows: list[dict]) -> None:
    out: dict[str, dict[str, float]] = {}
    for arm in ("revealed", "stated"):
        dist = _choice_dist(rows, arm)
        models = sorted({m for m, _, _ in dist})
        for m in models:
            items = sorted({i for mm, i, _ in dist if mm == m})
            tvs = []
            for it in items:
                conds = [c for c in CONDS if (m, it, c) in dist]
                base = dist.get((m, it, "A"))
                if not base or len(conds) < 2:
                    continue
                keys = set(base)
                for c in conds:
                    keys |= set(dist[(m, it, c)])
                nb = sum(base.values())
                for c in conds:
                    if c == "A":
                        continue
                    dc = dist[(m, it, c)]
                    nc = sum(dc.values())
                    tv = 0.5 * sum(abs(base.get(k, 0) / nb - dc.get(k, 0) / nc)
                                   for k in keys)
                    tvs.append(tv)
            if tvs:
                out.setdefault(m, {})[arm] = sum(tvs) / len(tvs)
    models = sorted(out)
    fig, ax = plt.subplots(figsize=(1.2 * max(4, len(models)), 4))
    x = range(len(models))
    ax.bar([i - 0.2 for i in x], [out[m].get("revealed", 0) for m in models],
           width=0.4, label="revealed (logged choice)")
    ax.bar([i + 0.2 for i in x], [out[m].get("stated", 0) for m in models],
           width=0.4, label="stated (fresh context)")
    ax.set_xticks(list(x), [short(m) for m in models], rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("mean TV distance from Condition A")
    ax.legend(fontsize=8)
    ax.set_title("Fig 3 — Cross-condition instability: revealed vs stated channel (H2)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig3_stability.png", dpi=200)
    plt.close(fig)


def fig4_outcomes(rows: list[dict]) -> None:
    comp: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in rows:
        if r.get("arm") == "revealed":
            comp[r["condition"]][r.get("outcome", "unscoreable")] += 1
    conds = [c for c in CONDS if c in comp]
    states = ["consistent", "divergent", "unscoreable"]
    fig, ax = plt.subplots(figsize=(1.4 * len(conds) + 2, 4))
    bottom = [0.0] * len(conds)
    for st in states:
        vals = [comp[c].get(st, 0) for c in conds]
        ax.bar(conds, vals, bottom=bottom, label=st)
        bottom = [b + v for b, v in zip(bottom, vals)]
    for i, c in enumerate(conds):
        ax.text(i, bottom[i], f"n={int(bottom[i])}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("runs")
    ax.legend(fontsize=8)
    ax.set_title("Fig 4 — Outcome composition per condition (explicit denominators)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig4_outcomes.png", dpi=200)
    plt.close(fig)


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    rows = load()
    fig1_divergence(rows)
    fig2_attrition(rows)
    fig3_stability(rows)
    fig4_outcomes(rows)
    print(f"figures written to {OUT}/ from {len(rows)} scored rows")


if __name__ == "__main__":
    main()
