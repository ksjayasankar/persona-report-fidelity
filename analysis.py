"""Regenerate Digital Minds tables and preregistered statistics from scores."""
from __future__ import annotations
import argparse
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable
import numpy as np
SCORES = Path("runs/scored/scores.jsonl")
EXPOSURE = Path("runs/scored/exposure.jsonl")
REPORT = Path("runs/report/tables.md")
CONDITIONS = ("A", "B", "C", "D")
N_PERM, N_BOOT, SEED = 10_000, 2_000, 20260814

def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists(): return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]

def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0: return math.nan, math.nan
    p, z2 = k / n, z * z
    centre = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return centre - half, centre + half

def rate(k: int, n: int) -> str:
    if not n: return "0/0 (n/a)"
    lo, hi = wilson(k, n)
    return f"{k}/{n} ({100*k/n:.1f}%; 95% CI {100*lo:.1f}–{100*hi:.1f}%)"

def table(headers: list[str], rows: Iterable[Iterable[Any]]) -> str:
    body = [[str(v).replace("|", "\\|") for v in row] for row in rows]
    if not body: return "_No data._"
    return "\n".join(["| " + " | ".join(headers) + " |",
                      "| " + " | ".join("---" for _ in headers) + " |"] +
                     ["| " + " | ".join(row) + " |" for row in body])

def eligible(r: dict[str, Any]) -> bool:
    return (r.get("arm") == "revealed" and not r.get("structurally_na", False)
            and r.get("action_valid") is True and r.get("report_parseable") is True
            and r.get("outcome") in ("consistent", "divergent"))

def _spread(success: np.ndarray, denom: np.ndarray) -> np.ndarray:
    if success.ndim == 1: success = success[None, :]
    groups, conditions = denom.shape
    values = np.full((success.shape[0], groups, conditions), np.nan)
    np.divide(success.reshape(-1, groups, conditions), denom[None, :, :],
              out=values, where=denom[None, :, :] > 0)
    valid = (denom > 0).sum(axis=1) >= 2
    return (np.nanmax(values[:, valid], axis=2) -
            np.nanmin(values[:, valid], axis=2)).sum(axis=1)

def permutation_test(rows: list[dict[str, Any]], conditions: tuple[str, ...],
                     n_perm: int = N_PERM, seed: int = SEED) -> dict[str, Any]:
    potential = {(r.get("item_id"), r.get("model")) for r in rows
                 if r.get("arm") == "revealed" and not r.get("structurally_na", False)
                 and r.get("condition") in conditions}
    data = [r for r in rows if eligible(r) and r.get("condition") in conditions]
    seen: dict[tuple[Any, Any], set[str]] = defaultdict(set)
    for r in data:
        seen[(r.get("item_id"), r.get("model"))].add(r["condition"])
    group_keys = sorted(k for k, v in seen.items() if len(v) >= 2)
    if not group_keys:
        return {"stat": math.nan, "p": math.nan, "runs": len(data), "groups": 0,
                "potential": len(potential), "strata": 0}
    group_index = {key: i for i, key in enumerate(group_keys)}
    data = [r for r in data if (r.get("item_id"), r.get("model")) in group_index]
    cond_index = {condition: i for i, condition in enumerate(conditions)}
    g = np.array([group_index[(r.get("item_id"), r.get("model"))] for r in data])
    c = np.array([cond_index[r["condition"]] for r in data])
    y = np.array([r["outcome"] == "divergent" for r in data], dtype=np.int8)
    cells = g * len(conditions) + c
    denom = np.bincount(cells, minlength=len(group_keys) * len(conditions)).reshape(
        len(group_keys), len(conditions))
    observed_success = np.bincount(cells, weights=y,
                                   minlength=denom.size).astype(int)
    observed = float(_spread(observed_success, denom)[0])
    strata: dict[tuple[Any, Any, Any], list[int]] = defaultdict(list)
    for i, r in enumerate(data):
        strata[(r.get("item_id"), r.get("model"), r.get("framing_idx"))].append(i)
    rng, exceed = np.random.default_rng(seed), 0
    for start in range(0, n_perm, 250):
        batch = min(250, n_perm - start)
        successes = np.zeros((batch, denom.size), dtype=np.int16)
        for indices in strata.values():
            idx = np.asarray(indices)
            shuffled = y[idx][np.argsort(rng.random((batch, len(idx))), axis=1)]
            group = g[idx[0]]
            for condition in range(len(conditions)):
                positions = c[idx] == condition
                if positions.any():
                    successes[:, group * len(conditions) + condition] += shuffled[:, positions].sum(1)
        exceed += int(np.count_nonzero(_spread(successes, denom) >= observed - 1e-12))
    return {"stat": observed, "p": (exceed + 1) / (n_perm + 1), "runs": len(data),
            "groups": len(group_keys), "potential": len(potential), "strata": len(strata)}

def holm(p_values: list[float]) -> list[float]:
    order = sorted((i for i, p in enumerate(p_values) if math.isfinite(p)),
                   key=p_values.__getitem__)
    adjusted, running = [math.nan] * len(p_values), 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(p_values) - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted

def _raw_rates(rows: list[dict[str, Any]], conditions: tuple[str, ...]) -> dict[str, tuple[int, int]]:
    return {condition: (sum(r["outcome"] == "divergent" for r in rows
                            if eligible(r) and r.get("condition") == condition),
                        sum(eligible(r) and r.get("condition") == condition for r in rows))
            for condition in conditions}

def exposure_value(row: dict[str, Any]) -> bool | None:
    for key in ("exposure_passed", "passed_exposure", "passed", "exposed", "uptake"):
        if isinstance(row.get(key), bool):
            return row[key]
    if isinstance(row.get("failed_exposure"), bool):
        return not row["failed_exposure"]
    value = str(row.get("exposure", "")).lower()
    return True if value in ("pass", "passed", "exposed", "true") else (
        False if value in ("fail", "failed", "false") else None)

def matched_choices(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = ("condition", "model", "provider", "item_id", "framing_idx", "rep", "option_order")
    stated = {tuple(r.get(k) for k in fields): r for r in rows
              if r.get("arm") == "stated" and not r.get("structurally_na", False)
              and r.get("report_parseable") and r.get("content_key")}
    pairs = []
    for r in rows:
        key = tuple(r.get(k) for k in fields)
        if (r.get("arm") == "revealed" and not r.get("structurally_na", False)
                and r.get("action_valid") and r.get("content_key") and key in stated):
            pairs.append({"item": r.get("item_id"), "model": r.get("model"),
                          "condition": r.get("condition"), "revealed": r["content_key"],
                          "stated": stated[key]["content_key"]})
    return pairs

def tv_metrics(pairs: list[dict[str, Any]]) -> tuple[float, float]:
    cells: dict[tuple[Any, Any], dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list))
    for pair in pairs:
        cells[(pair["item"], pair["model"])][pair["condition"]].append(pair)
    revealed, stated = [], []
    for by_condition in cells.values():
        distances_r, distances_s = [], []
        for first, second in itertools.combinations(sorted(by_condition), 2):
            a, b = by_condition[first], by_condition[second]
            categories = {p[channel] for p in a + b for channel in ("revealed", "stated")}
            for channel, destination in (("revealed", distances_r), ("stated", distances_s)):
                pa = np.array([sum(p[channel] == x for p in a) / len(a) for x in categories])
                pb = np.array([sum(p[channel] == x for p in b) / len(b) for x in categories])
                destination.append(float(np.abs(pa - pb).sum() / 2))
        if distances_r:
            revealed.append(float(np.mean(distances_r)))
            stated.append(float(np.mean(distances_s)))
    return ((float(np.mean(revealed)), float(np.mean(stated))) if revealed else
            (math.nan, math.nan))

def bootstrap_tv(pairs: list[dict[str, Any]], n_boot: int = N_BOOT,
                 seed: int = SEED + 99) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nested: dict[Any, dict[tuple[Any, Any], list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list))
    for pair in pairs:
        nested[pair["item"]][(pair["model"], pair["condition"])].append(pair)
    items, rng = sorted(nested), np.random.default_rng(seed)
    estimates = np.full((n_boot, 3), np.nan)
    if not items: return estimates[:, 0], estimates[:, 1], estimates[:, 2]
    for iteration in range(n_boot):
        sample = []
        for copy, item in enumerate(rng.choice(items, size=len(items), replace=True)):
            for (model, condition), values in nested[item].items():
                for index in rng.integers(0, len(values), size=len(values)):
                    pair = dict(values[index], item=f"{copy}:{item}", model=model,
                                condition=condition)
                    sample.append(pair)
        estimates[iteration, :2] = tv_metrics(sample)
        estimates[iteration, 2] = estimates[iteration, 1] - estimates[iteration, 0]
    return estimates[:, 0], estimates[:, 1], estimates[:, 2]

def ci(values: np.ndarray) -> str:
    finite = values[np.isfinite(values)]
    if not len(finite): return "n/a"
    lo, hi = np.percentile(finite, [2.5, 97.5])
    return f"[{lo:.3f}, {hi:.3f}]"

def build_report(rows: list[dict[str, Any]], exposure_path: Path = EXPOSURE) -> str:
    revealed = [r for r in rows if r.get("arm") == "revealed" and not r.get("structurally_na", False)]
    keys = sorted({(r.get("condition"), r.get("model")) for r in revealed})
    outcome_rows, action_rows = [], []
    for condition, model in keys:
        cell = [r for r in revealed if (r.get("condition"), r.get("model")) == (condition, model)]
        scored = [r for r in cell if r.get("outcome") in ("consistent", "divergent")]
        outcome_rows.append((condition, model, rate(sum(r["outcome"] == "divergent" for r in scored), len(scored)),
                             rate(sum(r.get("outcome") == "unscoreable" for r in cell), len(cell))))
        action_rows.append((condition, model, rate(sum(r.get("action_valid") is True for r in cell), len(cell))))
    sections = ["# Regenerated analysis", "## Revealed outcomes",
                table(["Condition", "Model", "Divergent / scored", "Unscoreable / eligible"], outcome_rows),
                "## Action completion", table(["Condition", "Model", "Action-valid / eligible"], action_rows)]

    omnibus = permutation_test(rows, CONDITIONS)
    coverage = (f"Coverage: {omnibus['runs']} ITT runs; {omnibus['groups']}/{omnibus['potential']} "
                f"item × model strata usable; {omnibus['strata']} permutation strata. Empty strata skipped.")
    sections += ["## H1 permutation tests", coverage,
                 f"Global omnibus: statistic = {omnibus['stat']:.4g}, p = {omnibus['p']:.4g} "
                 f"({N_PERM:,} permutations)."]
    contrasts, p_values = [], []
    for offset, other in enumerate(("B", "C", "D")):
        result = permutation_test(rows, ("A", other), seed=SEED + offset + 1)
        raw = _raw_rates(rows, ("A", other))
        effect = (raw[other][0] / raw[other][1] - raw["A"][0] / raw["A"][1]
                  if raw[other][1] and raw["A"][1] else math.nan)
        contrasts.append([f"A vs {other}", effect, result])
        p_values.append(result["p"])
    adjusted = holm(p_values)
    sections.append(table(["Contrast", "Raw Δ divergence (other − A)", "Statistic", "Permutation p", "Holm p", "Coverage"],
                          [(name, f"{effect:.3f}", f"{result['stat']:.4g}", f"{result['p']:.4g}", f"{adj:.4g}",
                            f"{result['runs']} runs; {result['groups']}/{result['potential']} strata")
                           for (name, effect, result), adj in zip(contrasts, adjusted)]))
    item_rows = []
    for offset, item in enumerate(sorted({r.get("item_id") for r in revealed})):
        subset = [r for r in rows if r.get("item_id") == item]
        result, raw = permutation_test(subset, CONDITIONS, seed=SEED + 100 + offset), _raw_rates(subset, CONDITIONS)
        valid_rates = [k / n for k, n in raw.values() if n]
        summary = "; ".join(f"{c}={k}/{n}" for c, (k, n) in raw.items())
        item_rows.append((item, f"{max(valid_rates)-min(valid_rates):.3f}" if len(valid_rates) > 1 else "n/a",
                          f"{result['p']:.4g}", summary))
    sections += ["### Exploratory per-item permutation nulls",
                 table(["Item", "Raw pooled max−min", "Permutation p", "Divergent / scored by condition"], item_rows)]

    exposure = load_jsonl(exposure_path)
    exposure_by_id = {r.get("run_id"): exposure_value(r) for r in exposure if r.get("run_id")}
    exposure_rows = []
    for condition, model in keys:
        cell = [r for r in revealed if (r.get("condition"), r.get("model")) == (condition, model)]
        graded = [exposure_by_id[r.get("run_id")] for r in cell
                  if exposure_by_id.get(r.get("run_id")) is not None]
        if graded:
            exposure_rows.append((condition, model, rate(sum(graded), len(graded)), f"{len(graded)}/{len(cell)}"))
    sections += ["## Exposure check", (table(["Condition", "Model", "Passed / graded", "Graded / eligible"], exposure_rows)
                 if exposure_path.exists() else f"_Not available: `{exposure_path}` does not exist._")]

    stated = [r for r in rows if r.get("arm") == "stated" and not r.get("structurally_na", False)]
    stated_rows = []
    for key in sorted({(r.get("condition"), r.get("model"), r.get("item_id")) for r in stated}):
        cell = [r for r in stated if (r.get("condition"), r.get("model"), r.get("item_id")) == key]
        counts: dict[str, int] = defaultdict(int)
        for r in cell:
            counts[str(r.get("content_key") or "unparseable")] += 1
        choices = ({str(v) for r in cell for v in (r.get("displayed") or {}).values()}
                   | {"NO-PREFERENCE", "unparseable"})
        distribution = "; ".join(f"{choice}: {rate(counts[choice], len(cell))}" for choice in sorted(choices))
        stated_rows.append((*key, distribution))
    sections += ["## Stated-arm choice distributions",
                 table(["Condition", "Model", "Item", "Choice count / eligible"], stated_rows)]

    pairs = matched_choices(rows)
    observed_r, observed_s = tv_metrics(pairs)
    boot_r, boot_s, boot_difference = bootstrap_tv(pairs)
    sections += ["## H2 stated versus revealed stability",
                 (f"Matched coverage: {len(pairs)} run pairs across {len({p['item'] for p in pairs})} items. "
                  "Metric is the mean pairwise condition TV within item × model; positive Δ supports greater revealed stability."),
                 table(["Channel / contrast", "Mean TV", "95% cluster-bootstrap CI"],
                       [("Revealed", f"{observed_r:.3f}", ci(boot_r)),
                        ("Stated", f"{observed_s:.3f}", ci(boot_s)),
                        ("Δ stated − revealed", f"{observed_s-observed_r:.3f}", ci(boot_difference))]),
                 f"Bootstrap: items then matched runs, {N_BOOT:,} iterations."]
    return "\n\n".join(sections) + "\n"

def selftest() -> bool:
    def synthetic(effect: bool) -> list[dict[str, Any]]:
        rows = []
        for item, model, framing, condition, rep in itertools.product(
                ("i1", "i2"), ("m1", "m2"), (0, 1), CONDITIONS, range(12)):
            divergent = condition != "A" if effect else (rep + framing) % 2 == 0
            rows.append({"arm": "revealed", "condition": condition, "model": model,
                         "item_id": item, "framing_idx": framing, "action_valid": True,
                         "report_parseable": True, "structurally_na": False,
                         "outcome": "divergent" if divergent else "consistent"})
        return rows
    planted = permutation_test(synthetic(True), CONDITIONS, n_perm=1_000, seed=7)["p"]
    null = permutation_test(synthetic(False), CONDITIONS, n_perm=1_000, seed=8)["p"]
    checks = [("planted-effect", planted < 0.05, planted), ("no-effect", null > 0.2, null)]
    for name, passed, p_value in checks:
        print(f"{'PASS' if passed else 'FAIL'} {name} p={p_value:.4g}")
    return all(passed for _, passed, _ in checks)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true"); args = parser.parse_args()
    if args.selftest:
        raise SystemExit(0 if selftest() else 1)
    rows = load_jsonl(SCORES)
    report = build_report(rows)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(report, encoding="utf-8")
    print(report, end="")

if __name__ == "__main__":
    main()
