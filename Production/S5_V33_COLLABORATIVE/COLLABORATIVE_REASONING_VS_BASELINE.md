# Does reasoning change the collaborative result? Qwen3:8B, three arms

Machine-readable form: `s5_16m_collaborative_reasoning_comparison.json`.

The published Qwen3:8B collaborative row ran with thinking **off**, because
`stronger_ai.py:619` hardcodes `'think': False`. This measures what reasoning changes.

## Why there are two new arms and not one

Reasoning on its own is unusable here. With `think: true` and the package's schema untouched,
qwen3:8b thinks at length and then returns `leading_rules: {}` — **0 of 6 rule families in 8 of 8
sampled turns**. `collaborative_qahs_s5.py:39-63` returns `None` unless all six families parse, and
every branch that could favour the LLM is gated on `llm_world is not None`, so the arbiter would
never even consider the LLM and the condition would silently collapse into plain QAHS.

The cause is that `leading_rules` carries no `required` list and no `minItems`, so `{}` is
grammar-legal. The non-thinking model fills it voluntarily; the thinking model spends its budget in
the reasoning block and emits the minimum legal object.

Requiring the six families restores it (6/6). But that is a second change, so it gets its own
control arm:

| Arm | `think` | Schema | Purpose |
|---|---|---|---|
| baseline (published) | false | as shipped | the canonical row |
| **C1** | false | families required | isolates the schema change |
| **B** | **true** | families required | isolates reasoning on top of C1 |

No package file was edited. Both rewrites are applied in flight by `tools/reasoning_proxy.py`,
which logs every call, and `run_s5_benchmark.py --ollama-url` is pointed at it.

## Results

All three arms share `manifest_sha256 = 59e6e3b2…`, `metric_spec_sha256 = d9d0afba…` and
`profile: paper`, so they are paired on the same frozen benchmark and the same 18 episodes.

| Arm | G | A | Au | T | R | SAI |
|---|---|---|---|---|---|---|
| baseline | 0.000 | 1.000 | 0.708 | 0.500 | 0.000 | **0.4416** |
| C1 — schema only | 0.000 | 1.000 | 0.708 | 0.500 | 0.000 | **0.4416** |
| B — reasoning | 0.000 | 1.000 | **0.767** | **0.333** | 0.000 | **0.4200** |

**C1 is score-identical to the baseline.** Tightening the schema lifted family completeness from
~90% to 198/198 and shifted the arbiter (`qa_hs` 186→181, `diagnostic` 9→14) but moved no score. So
everything in arm B is attributable to reasoning, which is what the control was for.

### Reasoning vs the control, paired per episode

| Domain | C1 | B | Δ | n | B better | C1 better | ties | p |
|---|---|---|---|---|---|---|---|---|
| G | 0.000 | 0.000 | +0.000 | 5 | 0 | 0 | 5 | 1.000 |
| A | 1.000 | 1.000 | +0.000 | 3 | 0 | 0 | 3 | 1.000 |
| Au | 0.708 | 0.767 | **+0.059** | 5 | 2 | 0 | 3 | 0.500 |
| T | 0.500 | 0.333 | **−0.167** | 3 | 0 | 1 | 2 | 1.000 |
| R | 0.000 | 0.000 | +0.000 | 3 | 0 | 0 | 3 | 1.000 |
| **SAI** | 0.4416 | 0.4200 | **−0.0216** | | | | | |

Nothing is significant; the lowest p is 0.500. The two movements:

```
Au per episode  C1: [0.833, 0.778, 0.511, 0.417, 1.000]
                B : [0.833, 0.778, 0.667, 0.556, 1.000]   two episodes improved
T  per pair     C1: [0.500, 0.500, 0.500]
                B : [0.500, 0.500, 0.000]                 T03 collapsed
```

Autonomy is `(novel_experiment_rate + (1−redundant_action_rate) + (1−no_change_action_persistence))/3`,
so a model that deliberates before acting plausibly repeats itself less — the direction is sensible.
The transfer loss is a single pair, which at n = 3 is noise-dominated.

## The headline: the arbiter still never chose the LLM

| Decision | baseline | C1 | B (reasoning) |
|---|---|---|---|
| `qa_hs` | 186 (93.9%) | 181 (91.4%) | 188 (91.7%) |
| `diagnostic` | 9 (4.5%) | 14 (7.1%) | 14 (6.8%) |
| `agreement` | 3 (1.5%) | 3 (1.5%) | 3 (1.5%) |
| `llm` | **0** | **0** | **0** |
| `llm_action` | **0** | **0** | **0** |
| total turns | 198 | 198 | 205 |

This is the stronger form of the finding already recorded for Opus and for the non-thinking Qwen
run. The LLM's action was **never** selected across 205 turns, *even when the model reasoned for a
mean of 8,236 characters per turn and supplied a complete six-family hypothesis every single time*.
The composite in `_candidate_score` weights log posterior at 78%, and a single LLM guess does not
outscore the posterior MAP world — reasoning does not change that.

Note Au still moved despite `llm`/`llm_action` being zero. The LLM's action is passed into
`qahs.choose_action(obs, llm_action)` as a hint and the diagnostic branch uses the hypothesis, so the
model influences the trajectory without the arbiter ever handing it the turn.

**Read this as a statement about the arbiter's scoring rule, not about reasoning quality.**

## Cost

| | C1 | B (reasoning) |
|---|---|---|
| Wall per call | 3.4 s | **30.8 s** (max 74.4 s) |
| Output tokens / call | 228 | **2,341** (max 6,000) |
| Thinking chars / call | 0 | **8,236** (max 26,797) |
| Total output tokens | 45,172 | 583,105 |
| Run wall time | 17.5 min | ~2 h across restarts |

Reasoning costs roughly **9× the wall time and 10× the output tokens** to lose 0.02 SAI. $0 either
way — everything is local.

## Two harness failures, and what they were

Both were defects in the experiment harness, not in the package or the model. Recorded because they
shaped the run:

1. **A `num_predict: 6000` cap I added as a "safety valve" caused a hard failure.** The cap bounds
   thinking and content *together*: one call produced a 26,797-character think and had zero budget
   left for the JSON, so `content` came back empty and the package's unguarded `json.loads` aborted
   the suite at episode 12 of 18. 132 of the 133 calls to that point were clean. Healthy calls sit
   at p50 2,332 / p99 4,423 output tokens, so 6,000 only ever caught the pathological tail.
2. **Removing the cap entirely exposed a stall.** A later call returned nothing at all for the full
   900 s socket timeout. That is the second such event in ~140 calls (one occurred during probing
   and did not reproduce on four retries) and it is the signature of the repetition loop Qwen's
   guidance warns about for **greedy decoding in thinking mode**. An earlier note in this work
   claiming "no repetition loop at temperature 0" was based on only 12 probe calls and was too
   confident.

Final configuration: cap at 12,000 (clear of the p99 by ~3×), socket timeout 240 s, and up to two
logged retries covering stalled, truncated and empty responses. After hardening, the remaining 7
episodes completed with **0 retries and 0 failures**.

Temperature was kept at 0 to match the baseline exactly. Given the stall evidence, Qwen's
recommended thinking sampling (temp 0.6 / top_p 0.95 / top_k 20) is the documented fix and would be
the better choice for any longer reasoning run — at the cost of changing a second variable.

## What this does not say

- **Not significant.** 18 episodes per condition, lowest p = 0.500. The SAI difference rests on one
  transfer pair and two autonomy episodes.
- **Not reproducible.** The adapter sends no `seed`; thinking length varies run to run, and
  non-determinism on this GPU was measured directly earlier in this work.
- **A deviation from the frozen package.** Both arms change the request schema. They are experiments,
  **not** replacements for the published row — `s5_16m_collaborative_report.json` at the package root
  remains the canonical Qwen3:8B collaborative result.
- **No quantum-advantage claim.** The six 4-qubit QAOA registers remain uncoupled, so the cost
  function is classically exact.
- G = 0.0 and R = 0.0 in every arm: no generalization or robustness episode was solved by any
  configuration.
