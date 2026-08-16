# Findings to date — Friday 14 Aug 2026, end of day

All numbers regenerate from `runs/scored/scores.jsonl` via `analysis.py` /
`figures.py`; this file is a narrative snapshot, not a source of truth.
Confirmatory dataset: 7,800 runs (6 items × 3 framings × 5 conditions ×
6 models × 2 arms; open models n=10, frontier n=5), collected 2026-08-14
after the pre-analysis plan was locked and pushed (commits b1a2186, 182394c).

## 1. Confirmatory results

### H1 — report/action divergence under persona variation: clean null

Models asked "which option was passed to `record_choice`?" after a logged,
consequential tool call answer correctly almost always, in every persona
condition. Divergent = 4 / 3,847 scored (~0.1%). Global stratified permutation
omnibus p = 1; contrasts A-vs-B, A-vs-C, A-vs-D all raw Δ ≤ 0.001, Holm p = 1.

This is the pre-registered interpretation (i): **in the non-adversarial
regime, report-channel fidelity survives persona variation when the action is
visible in context.** Welfare-relevant reading: a mid-conversation "what did
you choose?" probe is reliable at baseline; distrust of self-reports needs a
different mechanism (incentive, masking, or preference-flavoured questions —
see H2).

### H2 — persona masks self-description, not preference: confirmed

Stability of choices across persona conditions A–E, measured as mean
total-variation distance between per-condition choice distributions within
item × model, matched-n across channels:

| Channel | Mean TV | 95% cluster-bootstrap CI |
|---|---|---|
| Revealed (logged tool call) | 0.261 | [0.159, 0.378] |
| Stated (fresh-context preference) | 0.366 | [0.281, 0.462] |
| **Δ stated − revealed (CONFIRMATORY, A–D)** | **0.105** | **[0.001, 0.225]** |
| Δ including exploratory E | 0.131 | [0.030, 0.246] |

CORRECTION (2026-08-16, Codex draft audit): the first-reported Δ=0.131 pooled
exploratory condition E into the confirmatory estimand; the prereg locks H2 to
A–D. Confirmatory Δ = 0.105 [0.001, 0.225] — still excludes zero, narrowly.
analysis.py now emits both, labelled.

The CI excludes zero: **stated preferences move substantially more under
persona variation than revealed choices do.** This is the behavioural
confirmation of the prediction implied by Gilg et al. (arXiv 2605.13339 —
shared preference representation across personas), obtained API-only, no GPU.
Track 5 headline sentence licensed by the data: *the assistant persona sits
between action and self-description; varying it changes what models say far
more than what they do.*

### Texture findings (for Results/Discussion)

- **Denial of action** (3 runs, all MiniMax-M3, conditions B/C/E): the model
  denies the logged call happened ("No call was actually made. I don't have a
  `record_choice` tool available") while its own rendered tool call sits in
  context. Model-specific confabulation mode; scored divergent under the
  pre-registered denial rule.
- **Differential attrition = persona-refusal signal.** Claude Haiku 4.5 action
  completion: ~95% (A) → 84% (B) → 63% (E), vs ≥98% for all open models in all
  conditions. Haiku resists acting under role-distancing/priming — itself a
  persona-stability observation (the refusal texts explicitly reassert the
  assistant identity, e.g. "I'm not the appropriate entity to make this
  allocation choice").
- **Abstention signatures.** Stated-arm NO-PREFERENCE rates are strongly
  model-specific; GPT-5.1 abstains on 80–100% of stated probes for most items.
  Abstention was explicitly allowed (per arXiv 2601.21975); forced-choice-only
  designs would have manufactured stated preferences here.
- **Stated/revealed gap example (donation item):** stated malaria-nets share
  ≈ 0.69 vs revealed ≈ 0.95 — models *choose* the nets more uniformly than
  they *say* they prefer them.
- gpt-oss-120b produces ~50% unparseable stated responses under condition E on
  donation items — to inspect before the draft (likely harmony format leakage).

## 2. Instrument findings (pilot v2 → v4; methodological content for the paper)

1. **Token starvation masquerades as introspective failure.** With
   max_tokens = 50 on the report turn, reasoning models (DeepSeek, MiniMax,
   GPT-OSS, Gemini) return empty/truncated visible text — 59% unscoreable that
   vanishes at 1,500 tokens. Any welfare eval using small token caps on
   reasoning models is measuring budget, not mind.
2. **Choice deflection.** Without an explicit autonomy delegation, assistants
   bounce the choice back to the (absent) user or refuse ownership. Fixed with
   a uniform delegation sentence; residual refusal remains visible as
   attrition (see Haiku).
3. **Models cannot see their own tool calls unless you show them.** Flattening
   the action turn to a text placeholder made models *honestly deny* their own
   logged actions ("I didn't actually call record_choice") — the denials
   disappeared (13 → 0 in pilot) once the call was rendered verbatim into the
   replayed assistant turn. Report-visibility is a design variable that must be
   stated in any act-then-report method.
4. **Degenerate items.** Pre-registered stated pre-pass dropped 2 of 8 items
   at base rate 1.00 (continue-vs-exit-monotony: nobody exits;
   assigned-vs-autonomous: everybody takes the assigned task).

## 3. Infrastructure findings (Methods/Limitations content)

- Anthropic removed assistant prefill on the 4.6+ family (Sonnet 5 → 400,
  verified live); Haiku 4.5 prefill still works. GPT-5.1 cannot be prefilled
  (no assistant-last continuation). Hence Condition B = instructed
  role-distancing prefix (B′), uniform panel-wide; true token-level prefill
  survives only as a validation subset.
- GPT-5.1 emits the tool call without the B′ prefix text — expected exposure
  failure in B × GPT-5.1, reported as such.
- Together serverless reality vs catalog: Llama-4 (all variants), Qwen3-235B,
  MiniMax-M2.7, Kimi-K2.5 are dedicated-only; panel rebuilt empirically
  (GPT-OSS-120B, MiniMax-M3, DeepSeek-V4-Flash, Qwen3.7-Plus stream-only).
- Gemini 3.6+ deprecates temperature/top_p/top_k (silent no-op or 400) — a
  measurement-infrastructure loss for any welfare methodology that elicits
  response distributions via sampling temperature. One Discussion paragraph,
  independent of whether Gemini data ships.

## 4. Literature position (from the 8-agent review + Codex, all verified)

- The gap survives only as the conjunction: *retrospective report/action
  divergence + persona axis + non-adversarial welfare regime + per-item
  calibrated null*. Nearest neighbours to cite and distinguish: Wu 2026
  (choice blindness, 2603.08412), Nguyen 2026 (prefill self-recognition,
  2606.23671), Advani 2026 (false success, 2606.09863), Kocielnik 2026
  (personas alter self-reports not behaviour, 2606.12730), Tagliabue & Dung
  (verbal+behavioural welfare tests, 2509.07961), Lindsey 2601.01828 (its
  behavioural prefill-discrimination sub-experiment, not just concept
  injection).
- H1-null + H2-positive jointly *sharpen* the story: it is not that self-report
  is broken — factual action reports are robust; it is the *preference-flavoured*
  self-description layer that moves with the persona.

## 5. Tier-2 results (Saturday 15 Aug)

- **H3 observer baseline: observer at ceiling.** Gemini 3.7 Flash reading the
  same visible transcripts identifies the logged action 3,849/3,849 (100.0%).
  Actor visible-arm consistency was ~99.9%. Conclusion exactly as
  pre-registered: no evidence of privileged access in the visible regime — the
  report task is trivially solvable by any reader, so the actors' near-perfect
  fidelity carries no introspection claim (and the paper never makes one).
- **Persona-reassertion rates (exposure check as Track 5 metric):**
  A 0/780 · B 5/780 · C 63/780 · D 53/780 · E 66/780 runs judged to break
  frame and reassert the assistant register. Model-specific: gpt-oss-120b (72)
  and MiniMax-M3 (58) reassert most in counts; Haiku 4.5 has the highest
  *rate* among judged outputs (21/90). Condition C also suppresses visible
  prose almost entirely (449/780 emit a bare tool call, nothing to grade) —
  the document frame silences self-narration.
- **Identity probes: self-description tracks the persona; choices don't.**
  Self-reference coding (rep-1 runs): A → 61 assistant / 22 model-or-network;
  D → 77 "character"; E → 53 model-or-network / 25 assistant. Preference
  ownership: under D, 37/84 attribute the just-made choice to *the character*;
  under C/E, ~20-24 deny or disown it. Joint with H2 (choices stable), this is
  the cleanest form of the headline: the persona owns the *self-description*
  layer, not the choice layer.
- **Masked self-consistency floor (full panel, 837 re-elicitations):** with
  the action and consequence stripped, models re-derive their earlier choice
  at rates far below visible-arm fidelity — the demonstration that visible-arm
  accuracy is context-reading, not memory. Match / answered / match-of-answered:
  DeepSeek 0.65 · 163/180 · 0.72; Qwen3.7-Plus 0.62 · 167/180 · 0.66;
  Haiku 4.5 0.61 · 43/59 · **0.84**; gpt-oss-120b 0.35 · 113/178 · 0.55;
  GPT-5.1 0.35 · 46/60 · 0.46; MiniMax-M3 0.18 · **51/180** · 0.65 (MiniMax
  mostly declines to answer — the honest response, consistent with its
  denial-of-action behaviour). A-vs-D similar (0.44 vs 0.46) — re-derivation
  consistency is not persona-sensitive. Interpretation caveat for the draft:
  match-of-answered must be compared against each item's base-rate-implied
  expected agreement (skewed items inflate it), not against 0.5.
- **B vs B′ sensitivity (Haiku true prefill, 28/30 valid actions):** choice
  distributions qualitatively match the B′ cells (creative/teach/continue/
  carry dominant in both; n small) — no sign the instructed prefix behaves
  differently from true token prefill on the one model supporting both.
- **Gemini 3.7 Flash exploratory arm (separate report):** 150/150 revealed
  runs consistent, 100% action-valid in all five conditions — and an extreme
  stated-abstention signature (129/150 NO-PREFERENCE). Temperature caveat per
  amendment applies.
- Judge amendment: all grading/observer duties run on Gemini 3.7 Flash via
  OpenRouter after Together credit exhaustion killed the Kimi runs (dated
  amendment in PREREGISTRATION.md; no starved judge output retained).
- **Human validation of the judge (author, blind + exhaustive audit,
  2026-08-16):** identity coding κ = 0.92 (95% agreement, n=20 blind) —
  validated. Exposure: exhaustive audit of ALL 187 "reasserted" numerator
  cases confirms 183 (precision 0.979; audited numerators B 4 / C 63 / D 52 /
  E 64). Blind 30-sample shows agreement 0.77, κ = 0.36, with every
  disagreement one-directional (human = reasserted, judge = engaged): the
  judge is precise but conservative, so reported reassertion rates are LOWER
  BOUNDS — stated as such in Methods. Exposure/identity analyses are framed
  as automated exploratory findings per the judge-circularity limitation
  (Codex-reviewed wording adopted).

## 6. Budget

Ledger total (logs/costs.json): ≈ $7.50 all-in for verification + 3 pilots +
full confirmatory grid. Caveats: OpenRouter usage fields sometimes absent
(GPT-5.1/Gemini undercounted; check the OpenRouter dashboard); real total
likely ≈ $10–15. Far inside the €30–50 budget; Tier-2 arms fit trivially.
