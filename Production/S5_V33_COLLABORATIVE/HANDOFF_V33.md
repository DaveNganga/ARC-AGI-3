# Handoff: running the QAHS V3.3 collaborative S5 benchmark with Claude

For an agent or teammate picking this up in a new session. Scope: only the professor's latest
package, `QAHS_V33_COLLABORATIVE_S5_PRODUCTION_READY_TO_RUN.zip`, kept in this folder.

## Rules

- Never echo, log or commit an API key. Pass it via the environment only.
- Do not edit the professor's package files. The one approved change is the Anthropic adapter in
  `stronger_ai.py`, already applied. Every other file must match the zip (check 2 below).
- Run Claude only through `RUN_COLLABORATIVE_CLAUDE.sh`. Do not run `RUN_COLLABORATIVE_PAPER.sh`
  with `PROVIDER=anthropic`: it writes to Qwen's paths with `--resume` and would reuse Qwen's episodes.
- Do not cherry-pick: report every completed run, and name any run you exclude and why.
- Work on branch `claude/dazzling-goodall-4fc42q` of `abylAI02/ARC-AGI-3`. Commit results as you go.

## What V3.3 is

The S5 benchmark (16^6 = 16,777,216 hidden-rule worlds, 18 paper-profile episodes per condition)
with a third condition, `collaborative`, which arbitrates between the LLM's hypothesis and the QAHS
posterior. V3.3 leaves the V3.2 code paths unchanged, so the existing Claude Opus 5.5 V3.2 results
(`../S5_16M_STRONGER_AI_PRODUCTION/claude_opus_5_5/`) remain valid for `llm-only` and `qahs`; only
`collaborative` needs new runs.

| Model | LLM-only SAI | QAHS SAI | Collaborative SAI |
|---|---|---|---|
| Qwen3:8B (professor) | 0.400 | 0.475 | not in the zip; the professor runs it locally |
| Claude Opus 5.5 | 0.718 | 0.459 | 0.430 |

## Setup (about 2 minutes)

```bash
cd Production/S5_V33_COLLABORATIVE
pip install -r requirements.txt pytest
export ANTHROPIC_API_KEY=...        # or map from your own variable; never print it
```

Network: the run needs `api.anthropic.com` only; package registries for the install.

## Checks before spending anything (all free)

1. Environment: `python check_env.py`. numpy, scipy, qiskit and qiskit_aer must be OK. An
   Ollama failure is expected (Ollama is only for Qwen).
2. Package integrity: unzip the zip into an empty scratch directory and `cmp` each file. Only
   `stronger_ai.py` may differ; the extra files are `HANDOFF_V33.md`, `CLAUDE_RUN_NOTES_V33.md`,
   `RUN_COLLABORATIVE_CLAUDE.sh`, the zip itself and `claude_*/` results.
3. Adapter: `cmp stronger_ai.py ../S5_16M_STRONGER_AI_PRODUCTION/stronger_ai.py` must print
   nothing (same adapter as the Opus V3.2 run).
4. Tests: `python -m pytest -q test_s5_production.py test_collaborative_qahs_v33.py` → 8 passed.
5. Placeholder smoke with the quantum sampler (no API calls):
   ```bash
   python -u run_s5_benchmark.py --condition collaborative --profile smoke --experiment smoke \
     --provider placeholder --model placeholder --require-quantum --max-steps 12 \
     --output-dir /tmp/v33_smoke --report /tmp/v33_smoke/report.json
   ```
6. One real Claude smoke episode (about $0.02 on Opus): the same command with
   `--provider anthropic --model <model>`. Expect `solved: true` and the quantum sampler used.

## Run

```bash
bash RUN_COLLABORATIVE_CLAUDE.sh                          # Claude Opus 5.5 -> claude_opus_5_5/
MODEL=claude-haiku-5-5 bash RUN_COLLABORATIVE_CLAUDE.sh   # Claude Haiku 5.5 -> claude_haiku_5_5/
```

- Settings: identical to `RUN_COLLABORATIVE_PAPER.sh` (paper profile, 12 actions, 24-qubit
  QAOA, arbitration margin 0.15, diagnostic threshold 0.12). Effort defaults to medium
  (`ANTHROPIC_EFFORT`).
- Output: `<model>/collaborative_results/` (episode traces, prompt audits) and
  `<model>/s5_16m_collaborative_report.json`.
- Comparison: when the V3.2 Claude reports exist for the same model (Opus only today), the launcher
  checks that model, manifest, metric spec and profile match, then writes
  `<model>/s5_16m_collaborative_comparison.json` with paired sign tests.
- Order of work: generalization (5) → adaptation (3) → transfer (source + 3 cold/warm pairs) →
  robustness (3 clean/perturbed pairs).

| Model | Expected cost | Expected time |
|---|---|---|
| Claude Opus 5.5 ($4 / $20 per MTok) | about $5.50 (budget $8) | about 30-45 min |
| Claude Haiku 5.5 ($0.10 / $0.50 per MTok) | about $0.15 (budget $1) | shorter |

Estimate basis: the Opus V3.2 runs used 0.85-0.93M input and 0.10-0.11M output tokens per condition.

## If the run stops

- Any failed API call (credit, refusal, network) stops the run; no fake episode is recorded.
- Rerun the same command: `--resume` reuses finished episodes.
- The launcher deletes an incomplete transfer block first. This works around a bug in
  `run_s5_benchmark.py`, which reloads transfer state on resume only for `--condition qahs`.
- If credit ran out, top it up first. Name any partial run in the notes; do not delete it silently.

## After the run

1. Check the report: `SAI` and `scores` (G, A, Au, T, R), and the arbitration counts in each
   episode's `agent_summary.qahs.collaboration.counts`.
2. Commit everything under `<model>/` and push to the branch.
3. Report: the SAI table (LLM-only / QAHS / collaborative), the sign-test p-values from the
   comparison file, the token use and cost (`input_tokens`, `output_tokens` in episode summaries), and
   any failed or excluded runs.

## Caveats to state with the results

- 18 episodes per condition: one solve moves G by 0.2, so expect wide uncertainty.
- The arbiter uses the first listed value per rule family as the LLM's hypothesis, and only when
  all six families are present. Opus often omits family G (complete on 63% of V3.2 turns; Qwen
  90%, mostly the ontology listed in order). On other turns the arbiter falls back to QAHS.
- The QAOA cost has no couplings between the six 4-qubit registers, so it is classically exact and
  cheap; no quantum-advantage claim is possible.
- Claude cannot run at `temperature=0` as Qwen does, so its runs are not deterministic.
- More detail: `CLAUDE_RUN_NOTES_V33.md` (review findings) and
  `../S5_16M_STRONGER_AI_PRODUCTION/CLAUDE_RUN_NOTES.md` (the adapter fix).

## Status (2026-11-08)

- **Claude Opus 5.5 collaborative: complete.** 18 episodes, 204 calls, no failed calls,
  0.90M input and 0.10M output tokens, $5.63. Results in `claude_opus_5_5/`.

  | Opus 5.5 | G | A | Au | T | R | SAI |
  |---|---|---|---|---|---|---|
  | LLM-only (V3.2) | 0.200 | 0.944 | 0.944 | 0.500 | 1.000 | 0.718 |
  | QAHS (V3.2) | 0.000 | 1.000 | 0.754 | 0.542 | 0.000 | 0.459 |
  | Collaborative (V3.3) | 0.000 | 1.000 | 0.708 | 0.444 | 0.000 | 0.430 |

  No paired difference is significant (collaborative vs LLM-only: lowest p = 0.25, Au; vs QAHS:
  p = 1.0 in every domain). The arbiter chose the QAHS action on 189 of 204 turns, ran a diagnostic
  action on 11 and found agreement on 4; it never chose the LLM. Likely cause: the QAHS candidate
  is the posterior MAP world, which the LLM's hypothesis can rarely outscore on log posterior (78%
  of the composite), and the LLM's hypothesis was often incomplete. So for Opus, collaborative
  behaved almost like QAHS.
- Claude Haiku 5.5: not run.
- Qwen3:8B collaborative: not in the zip; ask the professor for his report.
