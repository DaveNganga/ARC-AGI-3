# S5 16.8M Stronger-AI Production Benchmark

This package compares the same LLM in two conditions over a full six-family hypothesis language:

- **LLM-only**: the LLM observes the environment, maintains its own hypotheses, and selects the action.
- **LLM + QAHS V3.2-S5**: the same LLM proposes an action and structured rule shortlist; QAHS performs factorized hypothesis search, 24-qubit quantum-assisted posterior sampling, and posterior-conditioned multi-step action selection.

The hidden rule world is

`H = P x G x M x T x I x S`, with 16 alternatives per family.

Therefore `|H| = 16^6 = 16,777,216 = 2^24` complete candidate worlds. The full space is represented implicitly; it is not materialized as 16.8 million Python objects.

## Scientific claim boundary

A successful experiment supports a claim that **QAHS augmentation improves the same LLM's measured reasoning behavior** across predefined Generalization, Adaptation, Autonomy, Transfer, and Robustness domains. It does **not** establish AGI/Strong AI and does **not** by itself establish quantum computational advantage. A matched classical-search (CAHS) control is the appropriate later quantum-specific ablation.

## What is frozen before the S5 run

- 96-rule ontology: 16 rules each for P/G/M/T/I/S.
- 24-bit `P|G|M|T|I|S` encoding.
- Five-domain metric specification in `stronger_ai_metric_spec_v1_1.json`.
- Benchmark worlds and action budget in `benchmark_manifest.json`.
- All clean benchmark worlds have an environment-verified witness solution within 12 actions.
- Same model, prompts, visual seeds, action budget, and environment instances are used for both conditions.
- The active hidden world and true symbol/class mapping are never supplied to either agent.

`audit_s5_manifest.py` verifies these invariants before a run.


## QAHS V3.3 Collaborative Extension

This release also includes a separate collaborative condition. Use `RUN_COLLABORATIVE_PAPER.sh` for the V3.3 test. V3.3 does not replace V3.2: it adds peer-hypothesis arbitration and a diagnostic-experiment option while keeping the same hidden-world firewall and frozen S5 benchmark. See `COLLABORATIVE_QAHS_V33.md` and `README_QAHS_V33.md`.

## QAHS V3.2-S5 scale-out

Exact V3.2 evaluated all 256 S1 worlds. That exact representation is not feasible at 16^6. The S5 production scale-out retains the V3.2 architecture but uses:

1. a factorized Bayesian posterior over all six 16-valued families;
2. bounded joint particles for interaction reasoning;
3. a **24-qubit QAOA register sampler** (4 qubits per family) over the complete `2^24` encoding;
4. candidate symbol/class bindings as nuisance hypotheses, rather than using the hidden true binding;
5. Explore -> Goal-Discrimination -> Exploit action modes;
6. receding-horizon multi-step plan generation and re-evaluation.

See `PROVENANCE.md` for the exact lineage and limitations.

## Install

For the current local Qwen/Ollama environment:

```bash
python -m pip install -r requirements.txt
ollama list
```

The QAHS condition requires Qiskit Aer. The distributed run scripts use `--require-quantum`, so they fail rather than silently fall back to a classical sampler.

## 1. Validate the environment

```bash
bash CHECK_ENV.sh
```

This checks Ollama/Qwen, Qiskit/Aer, a 24-qubit MPS circuit, all frozen witnesses, and the unit suite.

## 2. Run the one-world smoke test first

```bash
bash RUN_SMOKE.sh
```

The smoke world has a one-action verified witness and is intended only to prove the end-to-end LLM/QAHS/quantum integration before the expensive benchmark. Do not use the smoke result in the paper's five-domain statistics.

## 3. Run the paper profile sequentially

Do not run both conditions concurrently on one GPU/Ollama instance.

```bash
bash RUN_LLM_ONLY_PAPER.sh
bash RUN_QAHS_PAPER.sh
bash RUN_COMPARE.sh
```

Both benchmark scripts use `--resume`. If a run is interrupted, rerun the same command and completed episode JSON files will be reused.

The **paper** profile uses:

- Generalization: 5 held-out worlds.
- Adaptation: 3 hidden-change episodes.
- Autonomy: derived from the 5 Generalization traces (no extra LLM calls).
- Transfer: 1 source episode + 3 matched cold/warm targets.
- Robustness: 3 perturbed episodes; their clean counterparts are reused from Generalization.

That is 18 episodes per condition rather than redundantly rerunning clean/source data.

Outputs:

- `s5_16m_llm_report.json`
- `s5_16m_qahs_report.json`
- `s5_16m_comparison.json`
- per-episode traces under `s5_16m_paper_results/`
- `.partial` reports during long runs

## 4. Optional extended profile

If runtime permits before submission:

```bash
bash RUN_LLM_ONLY_EXTENDED.sh
bash RUN_QAHS_EXTENDED.sh
LLM_REPORT=s5_16m_llm_extended_report.json \
QAHS_REPORT=s5_16m_qahs_extended_report.json \
OUT=s5_16m_extended_comparison.json bash RUN_COMPARE.sh
```

The extended profile uses 10 Generalization, 5 Adaptation, 5 Transfer, and 5 Robustness cases.

## Five domain scores

The final report contains:

- `G`: held-out solve rate.
- `A`: mean post-change recovery efficiency `max(0,1-L/6)`; unrecovered episodes receive `L=6`.
- `Au`: mean `(NER + (1-RAR) + (1-NCAP))/3` over Generalization episodes.
- `T`: frozen paired warm-vs-cold transfer score.
- `R`: perturbed/clean solve retention, with the production rule that zero clean capability yields `R=0` rather than a misleading perfect robustness score.
- `SAI`: equal-weight mean of the five domain scores.

The comparison script also reports paired direction counts and exact two-sided sign tests by domain.

## Use a different model/provider later

The benchmark keeps the provider/model configurable so students can run the same frozen experiment with other models without changing the benchmark logic.

### OpenAI

```bash
export OPENAI_API_KEY=...
PROVIDER=openai MODEL=<model-name> bash RUN_LLM_ONLY_PAPER.sh
PROVIDER=openai MODEL=<same-model-name> bash RUN_QAHS_PAPER.sh
```

The OpenAI adapter uses the Responses API.

### Anthropic

```bash
export ANTHROPIC_API_KEY=...
PROVIDER=anthropic MODEL=<model-name> bash RUN_LLM_ONLY_PAPER.sh
PROVIDER=anthropic MODEL=<same-model-name> bash RUN_QAHS_PAPER.sh
```

Always use the **same model and settings in the paired LLM-only and QAHS conditions**. Do not compare one provider/model against a different provider/model and attribute the difference to QAHS.

## Important paper wording

A defensible conclusion, if the data support it, is:

> Across a 16,777,216-world hypothesis space, augmenting the same weak-AI LLM with QAHS improved predefined measures of generalization, adaptation, autonomous experimentation, transfer, and robustness. Under the study's operational definition, the combined architecture therefore exhibited stronger reasoning behavior than the standalone weak-AI baseline.

Do not convert that into a claim that the system is AGI/Strong AI or that QAOA itself has demonstrated a computational quantum advantage.
