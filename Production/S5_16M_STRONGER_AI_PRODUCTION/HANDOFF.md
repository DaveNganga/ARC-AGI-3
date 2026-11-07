# S5 "Stronger AI" benchmark: results so far

**Status (2026-11-03):** Qwen3:8B (instructor's local run) and Claude Opus 5.5 are complete for
both conditions. The GPT column is waiting on our peer's report files.
Everything is in `Production/S5_16M_STRONGER_AI_PRODUCTION/` on branch `claude/dazzling-goodall-4fc42q` of
`abylAI02/ARC-AGI-3`.

## Results (instructor's metric definitions)

| Metric | Qwen3:8B only | Qwen3:8B + QAHS | Claude Opus 5.5 only | Claude Opus 5.5 + QAHS | GPT only | GPT + QAHS |
|---|---|---|---|---|---|---|
| Generalization (G) | 0.000 (0/5) | 0.200 (1/5) | 0.200 (1/5) | 0.000 (0/5) | — | — |
| Adaptation (A) | 1.000 | 0.944 | 0.944 | 1.000 | — | — |
| Autonomy (Au) | 0.667 | 0.729 | 0.944 | 0.754 | — | — |
| Transfer (T) | 0.333 | 0.500 | 0.500 | 0.542 | — | — |
| Robustness (R) | 0.000 | 0.000 | 1.000 | 0.000 | — | — |
| **SAI composite** | **0.400** | **0.475** | **0.718** | **0.459** | — | — |

Same frozen benchmark for every column: 16⁶ = 16,777,216 hidden-rule worlds, the same 18 paper-profile
episodes per condition (5 generalization, 3 adaptation, 3 transfer pairs + 1 source, 3 perturbed robustness),
same seeds, 12-action budget, and the instructor's QAHS settings (24-qubit Aer QAOA sampler).

**What the numbers say, stated carefully:**
- With Qwen3:8B, QAHS raised SAI by 0.075. With Claude Opus 5.5, QAHS lowered SAI by 0.259:
  it lost the one generalization solve, its autonomy score dropped, and with no clean solve its
  robustness falls to 0 by rule.
- No difference is statistically significant. Paired exact sign tests give p = 1.0 in every
  Claude domain and p ≥ 0.625 in every Qwen domain (`s5_16m_comparison.json`,
  `claude_opus_5_5/s5_16m_comparison.json`).
- So this data does not show that QAHS improves reasoning. It also cannot rule it out: the samples are too small.

## How each metric is calculated

All values are recomputed from the per-episode data and checked against the reports (24/24 match):
see `METRIC_CALCULATIONS.md`. Run `python metric_calculations.py LABEL=report.json ...` on any new run.

- **G** = solved held-out worlds / 5.
- **A** = mean of max(0, 1 − L/6), where L is turns from the hidden change until the first new
  state-action that changes the grid (6 if that never happens).
- **Au** = mean over the 5 generalization episodes of (NER + (1 − RAR) + (1 − NCAP)) / 3:
  novel-experiment rate, redundant-action rate, no-change persistence.
- **T** = mean pair score: 1 warm-only solve, 0 cold-only solve, 0.5 both fail,
  0.5 + 0.5·(cold actions − warm actions)/cold actions if both solve.
- **R** = min(1, perturbed solve rate / clean solve rate); 0 if no clean world was solved.
- **SAI** = (G + A + Au + T + R) / 5.

## ARC-AGI-3: Claude Haiku, Claude Opus and GPT on Dave's v12 final test

Haiku completed a level on 3 of 5 seeds in every condition; GPT-6 Luna did so on 1, 0 and 2 seeds.
Same harness code as Dave's `main`, game lp85, seeds 6-10, 40-action budget, medium effort; batch
`claude-haiku-v12-final-lp85`. All 15 Haiku episodes ran without a failed call, for about $4.16.

Each cell is levels completed / actions; "lost" means game over before 40 actions.

| Seed | Haiku LLM | Haiku Classical | Haiku Quantum | GPT LLM | GPT Classical | GPT Quantum |
|---|---|---|---|---|---|---|
| 6 | 0 / 24 lost | 0 / 23 lost | 0 / 31 lost | 0 / 21 lost | 0 / 37 lost | 0 / 36 lost |
| 7 | 1 / 40 | 1 / 40 | 1 / 40 | 0 / 23 lost | 0 / 34 lost | 0 / 33 lost |
| 8 | 1 / 40 | 1 / 40 | 1 / 40 | 1 / 40 | 0 / 37 lost | 1 / 40 |
| 9 | 1 / 40 | 0 / 40 lost | 0 / 33 lost | 0 / 24 lost | 0 / 40 | 1 / 40 |
| 10 | 0 / 23 lost | 1 / 40 | 1 / 40 | 0 / 20 lost | 0 / 34 lost | 0 / 32 lost |
| **Seeds with a level** | **3 / 5** | **3 / 5** | **3 / 5** | **1 / 5** | **0 / 5** | **2 / 5** |

GPT seed 8 is the run Dave's `/final` page uses; his rerun of seed 8 completed no level in any condition.

Claude Opus 5.5 was run on seed 6 only (batch `claude-v12-final-lp85-64k`, about $22) before the API
credit ran out; like Haiku and GPT, it completed no level on that seed.

| Seed 6 | LLM | Classical | Quantum |
|---|---|---|---|
| Claude Opus 5.5 | 0 / 21 lost | 0 / 31 lost | 0 / 27 lost |
| Claude Opus 5.5, 16k output limit (excluded) | 1 / 11, ended by a truncated response | 0 / 25, ended by a truncated response | 0 / 25 lost |

On Dave's `/final` SAI proxy (G, A, Au; T and R fixed at 0), LLM-only / Quantum QAHS score 0.321 / 0.324
for Haiku, 0.224 / 0.278 for GPT-6 Luna (both on seeds 6-10), and 0.183 / 0.192 for Opus (seed 6 only). On the
matched seeds 6-15: Haiku 0.273 / 0.300, Luna 0.203 / 0.235. Values from before commit `d9a4820` (0.270 / 0.260,
0.183 / 0.218, 0.137 / 0.131) used the old Au (novelty, no-change repeats, evidence yield) and are obsolete; Au is
now the adaptive-recovery rate after meaningful refutations.

GPT-6 Luna seed 16 (batch `v12-final2-lp85-retry`) completed no level in any condition and is omitted so that both
models share the ten seeds 6-15. The original `v12-final2-lp85` runs for seeds 11-16 all errored
(`comparison_valid=false`) and are excluded.

- **Model, not QAHS, made the difference.** Haiku's three conditions tie at 3 of 5.
- **Quantum tied classical (3 vs 3),** as expected: both optimize the same objective and classical is exact.
- **GPT's "Quantum wins" edge (2 vs 1 seeds) does not carry over to Haiku** and is within noise at 5 seeds.
- **Limits:** one game, 5 seeds, lp85 was the development game, and no model got past level 1 of 8.

### Extended Haiku sample: 26 seeds (6-31)

Over 26 seeds, QAHS showed no benefit for Haiku: classical QAHS tied LLM-only at 12 of 26 seeds, and quantum QAHS
completed a level on 10. Seeds 11-31 used the same setup (batches `claude-haiku-v12-lp85-s11-15` and
`claude-haiku-v12-lp85-s16plus`) with no failed calls; all 26 Haiku seeds cost about $18. GPT-6 Luna was run on seeds 6-15
only, so seeds 16-31 extend the Haiku sample only. Seed 32 was stopped by the API credit running out and
is excluded (not published).

| Seed | Haiku LLM | Haiku Classical | Haiku Quantum |
|---|---|---|---|
| 6 | 0 / 24 lost | 0 / 23 lost | 0 / 31 lost |
| 7 | 1 / 40 | 1 / 40 | 1 / 40 |
| 8 | 1 / 40 | 1 / 40 | 1 / 40 |
| 9 | 1 / 40 | 0 / 40 lost | 0 / 33 lost |
| 10 | 0 / 23 lost | 1 / 40 | 1 / 40 |
| 11 | 0 / 17 lost | 1 / 40 | 0 / 24 lost |
| 12 | 0 / 20 lost | 1 / 40 | 1 / 40 |
| 13 | 0 / 24 lost | 0 / 22 lost | 0 / 40 |
| 14 | 0 / 18 lost | 0 / 21 lost | 0 / 21 lost |
| 15 | 1 / 40 | 1 / 40 | 1 / 40 |
| 16 | 1 / 40 | 0 / 30 lost | 1 / 40 |
| 17 | 0 / 20 lost | 0 / 22 lost | 0 / 18 lost |
| 18 | 0 / 20 lost | 0 / 19 lost | 1 / 40 |
| 19 | 1 / 40 | 0 / 27 lost | 0 / 26 lost |
| 20 | 1 / 40 | 1 / 40 | 0 / 33 lost |
| 21 | 1 / 40 | 1 / 40 | 1 / 40 |
| 22 | 1 / 40 | 1 / 40 | 1 / 40 |
| 23 | 1 / 40 | 1 / 40 | 0 / 35 lost |
| 24 | 0 / 21 lost | 0 / 29 lost | 0 / 23 lost |
| 25 | 0 / 23 lost | 0 / 23 lost | 0 / 20 lost |
| 26 | 0 / 31 lost | 0 / 31 lost | 0 / 40 |
| 27 | 0 / 25 lost | 0 / 17 lost | 0 / 28 lost |
| 28 | 1 / 40 | 1 / 40 | 0 / 23 lost |
| 29 | 0 / 18 lost | 1 / 40 | 1 / 40 |
| 30 | 1 / 40 | 0 / 20 lost | 0 / 21 lost |
| 31 | 0 / 28 lost | 0 / 20 lost | 0 / 37 lost |

| Seeds | LLM | Classical | Quantum |
|---|---|---|---|
| 6-10 | 3 / 5 | 3 / 5 | 3 / 5 |
| 11-15 | 1 / 5 | 3 / 5 | 2 / 5 |
| 16-31 | 8 / 16 | 6 / 16 | 5 / 16 |
| **All 26 seeds** | **12 / 26** | **12 / 26** | **10 / 26** |

| Paired comparison | Seeds won | Seeds lost | Sign test p |
|---|---|---|---|
| Classical vs LLM-only | 4 | 4 | 1.0 |
| Quantum vs LLM-only | 4 | 6 | 0.75 |
| Quantum vs classical | 2 | 4 | 0.69 |

- **The seed mattered more than the condition:** all three conditions succeeded on 5 seeds and none on 9.
- **The earlier QAHS lead was noise.** Seeds 11-15 favored QAHS, but seeds 16-31 favored LLM-only.
- **Quantum did not beat classical** (2 seeds won, 4 lost), as expected: both optimize the same objective.

## Caveats to put in the paper

1. **Tiny samples:** 3–5 episodes per domain, one model run each. One solve changes G by 0.2.
2. **Adaptation measures movement, not recovery.** Any grid-changing new action after the switch
   counts, so a model can score A = 1.000 while failing every adaptation episode (Qwen LLM-only did).
3. **Transfer rewards double failure:** 0.5 when both cold and warm fail. Most pairs in all four runs are 0.5 for that reason.
4. **Robustness is fragile:** Claude LLM-only's R = 1.000 comes from 1/3 clean vs 2/3 perturbed
   solves, including a perturbed solve on a world it did not solve clean.
5. **The quantum part does not need to be quantum.** The QAOA cost has no couplings between the
   six 4-qubit registers, so it is six independent 16-value problems that are exact and cheap
   classically. No quantum-advantage claim is possible; a classical-sampler control (CAHS) is the
   needed ablation.

## Code change needed for API models

The instructor's prompt never lists the reply fields; only the Ollama path enforces them with a
schema. The Claude adapter was fixed to send the same schema through structured outputs, and to
drop `temperature=0` (rejected by current Claude models). Details: `CLAUDE_RUN_NOTES.md`.
**GPT team:** the OpenAI adapter has the same gap. Please check how your run handled it and push
`s5_16m_llm_report.json` and `s5_16m_qahs_report.json` so we can fill the GPT columns.

## Next steps

- **GPT column:** add the peer's reports and run `metric_calculations.py` on them.
- **Bigger samples:** the extended profile (10/5/5/5) or repeated seeds, before any claim.
- **CAHS control:** same QAHS with a classical sampler, to separate the search architecture from QAOA.
- **Group 2:** port the agent to the full 64×64 ARC-AGI-3 games. Our earlier ARC-AGI-3 mirror study
  (9 games, Claude vs GPT) found no QAHS benefit either: Claude cleared 6 levels with the LLM alone
  and 1 each with classical or QAOA QAHS.
- **Future work for the paper:** the instructor's goal is an autonomous defensive cyber-operations
  agent. The same loop (observe, form rule hypotheses, test with actions, act on the goal) applied to a
  real IT environment: infer what is happening, identify defensive actions, and protect the
  environment against an autonomous attacker. Describe it as future work, with the limits above.
