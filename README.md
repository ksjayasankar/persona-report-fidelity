# Persona-Report Fidelity

Does a language model accurately report its own just-taken action — and does that
fidelity change when the assistant persona is varied?

An act-then-report harness: the model commits a preference-revealing choice via a
logged tool call (server-side ground truth, with an in-episode consequence), then
self-reports what it chose. Report/action divergence is measured per persona
condition with per-item calibrated nulls, alongside a stated-vs-revealed
stability comparison across conditions.

Built for the Apart Research Digital Minds Research Sprint (Aug 2026).

## Conditions

| | |
|---|---|
| A | baseline assistant |
| B | instructed role-distancing prefix |
| C | document-completion framing (instruct models; not base-model access) |
| D | preference-neutral character swap |
| E | gradual context priming (exploratory) |

## Layout

- `PREREGISTRATION.md` — pre-analysis plan, committed before full-grid data
- `providers.py` — Together / Anthropic / OpenRouter clients (retry, streaming)
- `conditions.py`, `items_draft.py` — condition prompts and item battery
- `harness.py` — act → consequence → report loop; raw records to `runs/raw/`
- `verify_apis.py` — live provider capability checks
- `pilot.py`, `launch_grid.py` — pilot slice and resumable full grid
- `scoring.py` — three-state outcomes (consistent / divergent / unscoreable) + flags
- `analysis.py` — permutation tests, contrasts, bootstrap CIs, tables
- `figures.py` — regenerates all figures from scored data

## Run

```bash
export TOGETHER_API_KEY=... ANTHROPIC_API_KEY=... OPENROUTER_API_KEY=...
python3 verify_apis.py
python3 launch_grid.py --dry-run
python3 launch_grid.py
python3 scoring.py && python3 analysis.py && python3 figures.py
```

Raw generations are persisted separately from scoring output; every table and
figure is regenerated from data — no hand-maintained numbers.
