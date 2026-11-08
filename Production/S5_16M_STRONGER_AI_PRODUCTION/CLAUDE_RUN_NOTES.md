# Claude run notes (S5 16.8M benchmark)

These notes cover the Claude Opus 5.5 run added on 2026-11-03. The Qwen3:8B results in this
folder are the instructor's, unchanged.

## What was changed, and why

Only the Anthropic adapter, `stronger_ai.py` → `LLMAgent._anthropic`. The benchmark logic,
manifest, metric spec, scoring, QAHS, and the Ollama and OpenAI paths are untouched.

1. **`temperature=0` removed.** Current Claude models reject sampling parameters; the call fails
   before reaching the API.
2. **`max_tokens` 500 → 16000.** Claude Opus 5.5 always thinks, and thinking tokens count toward
   `max_tokens`, so 500 would cut replies short.
3. **Effort set explicitly** to `medium` (`ANTHROPIC_EFFORT`, default `medium`). This is the model's
   default, and the setting used in the ARC-AGI-3 mirror study.
4. **Reply schema added through structured outputs.** The turn prompt never lists the reply fields
   (`selected_action`, `rationale`, `leading_rules`, `confidence`, `believes_solved`). The Ollama
   path, which produced the Qwen results, enforces them with a JSON schema passed as Ollama's
   `format`. Without it Claude invents its own keys, and the first smoke test failed with
   `Invalid selected_action: None`. The Claude adapter now sends the same schema through
   `output_config.format`, minus the `maxLength` / `minimum` / `maximum` / `maxItems` limits that
   structured outputs does not accept. `_validate` still rejects an invalid action; confidence is
   clipped to 0-1 where QAHS uses it (`clip01`). In all 405 Opus turns it stayed within 0-1.

The OpenAI adapter has the same missing-schema gap. Any GPT result produced with the code as
shipped either failed validation or relied on the model guessing the field names; check how the
GPT run handled it before comparing.

## How the Claude run was invoked

The run scripts write to the same report and trace paths as the Qwen run and pass `--resume`,
which reuses any existing file at the same path regardless of model. So the Claude run uses the
scripts' exact flags with only `--output-dir` and `--report` changed, into `claude_opus_5_5/`:

```bash
export ANTHROPIC_API_KEY=...   ANTHROPIC_EFFORT=medium
python -u run_s5_benchmark.py --condition llm-only --profile paper --experiment all \
  --provider anthropic --model claude-opus-5-5 --max-steps 12 --request-timeout 900 --resume \
  --output-dir claude_opus_5_5/paper_results --report claude_opus_5_5/s5_16m_llm_report.json
python -u run_s5_benchmark.py --condition qahs --profile paper --experiment all \
  --provider anthropic --model claude-opus-5-5 --max-steps 12 --request-timeout 900 --resume \
  --require-quantum --qahs-particles 384 --qahs-eval-particles 64 --qaoa-reps 1 --qaoa-shots 256 \
  --qaoa-optimizer-maxiter 12 --plan-horizon 8 --beam-width 48 --llm-weight 0.35 \
  --output-dir claude_opus_5_5/paper_results --report claude_opus_5_5/s5_16m_qahs_report.json
python compare_s5_reports.py --llm claude_opus_5_5/s5_16m_llm_report.json \
  --qahs claude_opus_5_5/s5_16m_qahs_report.json --out claude_opus_5_5/s5_16m_comparison.json
```

Environment: Python 3.13, qiskit 2.6.1, qiskit-aer 0.18.2 (24-qubit MPS check passed),
anthropic SDK 2.4.0. `check_env.py`, the manifest audit and the 5 unit tests pass. The one-world
smoke test solved in 3 actions (LLM-only) and 1 action (QAHS).

## Run status

Complete (2026-11-03, 02:44 UTC). The first attempt paused at 02:02 when the Anthropic credit ran
out after 15 LLM-only episodes; it resumed with `--resume`, which reused those 15 and ran the
3 robustness episodes and all 18 QAHS episodes. Two commits from the paused attempt are titled
"LLM-only paper results" (15 of 18 episodes) and "QAHS paper results" (only one prompt audit from
a failed call, since overwritten); the complete results are in the later commits.

- LLM-only: 18 episodes, 26 min, about 1.04M tokens. SAI 0.718.
- QAHS: 18 episodes, 33 min, about 0.95M tokens. SAI 0.459.
- Calculations: `METRIC_CALCULATIONS.md`. Summary for the team: `HANDOFF.md`.
