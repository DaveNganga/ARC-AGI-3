# Claude on QAHS V3.3 Collaborative S5: setup and cost

The professor's V3.3 package (`QAHS_V33_COLLABORATIVE_S5_PRODUCTION_READY_TO_RUN.zip`, kept here),
with the same Anthropic adapter fix used for the Opus V3.2 run. Not yet run with Claude.

## What V3.3 changes relative to V3.2 (the package Qwen and Opus already ran)

- Adds the `collaborative` condition (`collaborative_qahs_s5.py`), its launchers and comparison script.
- `run_s5_benchmark.py` gains the condition and three arbitration flags.
- `qahs_s5.py` gains a public `_delta_likelihood` wrapper. V3.2's `QAHSV32S5` had no such method, so
  nothing in the `qahs` condition calls it: V3.2 behavior is unchanged.
- `benchmark_manifest.json`, `stronger_ai_metric_spec_v1_1.json`, `stronger_ai.py` and
  `compare_s5_reports.py` are byte-identical to V3.2. All files match `release_audit_v33.json`.

So the existing Opus LLM-only (SAI 0.718) and QAHS (0.459) reports stay valid; only the
collaborative condition needs a new run.

## The one change: the Anthropic adapter

`stronger_ai.py` is patched exactly as in `../S5_16M_STRONGER_AI_PRODUCTION/CLAUDE_RUN_NOTES.md`
(the file is byte-identical to the one the Opus V3.2 run used): no `temperature`, `max_tokens`
16000, effort medium, and the reply schema the Ollama path already sends Qwen. The schema matters
more in V3.3, because arbitration reads the LLM's `leading_rules` and `confidence`.

## Run

`RUN_COLLABORATIVE_PAPER.sh` must not be reused with `PROVIDER=anthropic`: it writes to Qwen's
paths with `--resume`, so it would reuse Qwen's episodes. `RUN_COLLABORATIVE_CLAUDE.sh` has the
same flags with Claude output paths, then builds the four-way comparison:

```bash
export ANTHROPIC_API_KEY=...
bash RUN_COLLABORATIVE_CLAUDE.sh                        # Opus 5.5 -> claude_opus_5_5/
MODEL=claude-haiku-5-5 bash RUN_COLLABORATIVE_CLAUDE.sh   # Haiku 5.5 -> claude_haiku_5_5/
```

Checked here: the 8 package tests pass, and a placeholder-model smoke run of `collaborative`
with the Aer QAOA sampler completes and records the same manifest and metric-spec hashes. An
offline mock of the Anthropic call (anthropic SDK 2.6.0; the Opus V3.2 run used 2.4.0, and both
send API version 2026-10-21) shows the exact request: no `temperature`, `max_tokens` 16000,
effort medium, the reply schema.

## Review findings (2026-11-08)

1. **Resume bug in `run_s5_benchmark.py` (fixed in the launcher, not in the package).** On
   `--resume`, line 248 reloads the transfer source's QAHS state only for `--condition qahs`. A
   collaborative run resumed between the transfer source and its warm episodes would run those
   warm episodes without the transferred posterior, which changes the Transfer score. Reproduced
   with the placeholder model: two fresh runs gave identical warm episodes, and the resumed run
   took different actions in all three. `RUN_COLLABORATIVE_CLAUDE.sh` deletes an incomplete
   transfer block before resuming, so it reruns in one pass as in a fresh run. An uninterrupted
   run, such as the professor's Qwen run, is unaffected. The Opus V3.2 QAHS run was not affected:
   it ran all 18 episodes in one pass.
2. **The comparison script does not check the model.** `compare_s5_collaborative.py` checks the
   benchmark and profile only, so it would pair a Qwen report with a Claude one. The launcher
   checks provider, model, manifest, metric spec and profile first.
3. **Confidence is not clamped in `_validate`.** Qwen's schema bounds it to 0-1; the Claude
   schema cannot. QAHS and the arbiter clip it with `clip01`, and in all 405 Opus V3.2 turns it
   stayed within 0-1.
4. **For interpreting results (professor's design; affects both models alike):**
   - The arbiter takes the first value listed per rule family as the LLM's hypothesis and uses it
     only when all six families are present. Qwen usually lists the ontology in order (`P0, P1, P2,
     P3`), so its "hypothesis" is mostly P0 G0 M0 T0 I0 S0. Opus lists real candidates but often
     omits family G, so in the V3.2 traces its hypothesis would be complete on 63% of turns
     (Qwen: 90%). On the other turns the arbiter falls back to QAHS.
   - `_candidate_predictive_fit` locates each history entry with `history.index(h)`, which finds
     the first equal entry, so a repeated identical move (for example bumping the same wall twice)
     is scored against the earlier history prefix. Minor: the term has 22% weight.

## Cost estimate (paper profile, 18 episodes)

Collaborative makes one LLM call per turn, like the `qahs` condition, so token use should match
the Opus V3.2 runs (measured from their traces):

| Opus V3.2 run | Input tokens | Output tokens | Cost at $4 / $20 per MTok |
|---|---|---|---|
| LLM-only | 927,399 | 108,426 | $5.88 |
| QAHS | 848,250 | 101,997 | $5.43 |
| **Collaborative (estimate)** | ~0.85-0.93M | ~0.10-0.11M | **~$5.50 (budget $8)** |

Haiku 5.5 at the same token counts: about $0.15 (budget $1). Opus wall time was 26-33 min per condition.
