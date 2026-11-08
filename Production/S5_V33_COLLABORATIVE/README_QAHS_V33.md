# QAHS V3.3 Collaborative S5 / 16M Package

This package adds a collaborative QAHS condition to the S5 16,777,216-world benchmark.

## Conditions

- `llm-only`: base LLM selects the executed action.
- `qahs`: frozen QAHS V3.2-S5 override architecture.
- `collaborative`: QAHS V3.3, which treats LLM and QAHS as peer hypotheses and arbitrates between them or requests a diagnostic experiment.

The benchmark, world manifest, action budget, ontology, information firewall, and V3.2 code are unchanged.

## Run a smoke test

```bash
python -u run_s5_benchmark.py --condition collaborative --profile smoke --experiment smoke --provider placeholder --model placeholder --no-quantum --output-dir smoke_collab --report smoke_collab_report.json
```

## Run the real paper experiment with local Qwen

```bash
bash CHECK_ENV.sh
bash RUN_COLLABORATIVE_PAPER.sh
```

The production script uses `--require-quantum` and will refuse to silently fall back if Qiskit Aer is unavailable.

## Run with Claude

Set the provider/model in the environment while keeping the same frozen benchmark and parameters:

```bash
export ANTHROPIC_API_KEY=...
PROVIDER=anthropic MODEL=<claude-model-name> bash RUN_COLLABORATIVE_PAPER.sh
```

## Compare collaboration to the existing runs

After the collaborative report is produced:

```bash
python compare_s5_collaborative.py \
  --llm s5_16m_llm_report.json \
  --baseline s5_16m_qahs_report.json \
  --collaborative s5_16m_collaborative_report.json \
  --out s5_16m_collaborative_comparison.json
```

## Scientific claim boundary

V3.3 is designed to test whether evidence-based collaboration can preserve the strengths of a stronger LLM while selectively using QAHS for hypothesis management, transfer, and diagnostic experimentation. A successful result is evidence for stronger reasoning behavior under the predefined five-domain operational definition; it is not evidence of AGI/Strong AI and does not establish quantum computational advantage.
