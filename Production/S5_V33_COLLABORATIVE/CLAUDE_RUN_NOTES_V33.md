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
with the Aer QAOA sampler completes and records the same manifest and metric-spec hashes.

## Cost estimate (paper profile, 18 episodes)

Collaborative makes one LLM call per turn, like the `qahs` condition, so token use should match
the Opus V3.2 runs (measured from their traces):

| Opus V3.2 run | Input tokens | Output tokens | Cost at $4 / $20 per MTok |
|---|---|---|---|
| LLM-only | 927,399 | 108,426 | $5.88 |
| QAHS | 848,250 | 101,997 | $5.43 |
| **Collaborative (estimate)** | ~0.85-0.93M | ~0.10-0.11M | **~$5.50 (budget $8)** |

Haiku 5.5 at the same token counts: about $0.15 (budget $1). Opus wall time was 26-33 min per condition.
