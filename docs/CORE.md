# QAHS — Core Document

This is the single, authoritative description of what this project is, why it exists, and what it has (and has not) shown. The assignment itself is in [`TASK.md`](TASK.md). Known problems and the roadmap are in [`IMPROVEMENTS.md`](IMPROVEMENTS.md). Operational how-tos live in `README.md`, `ARC_RUN_GUIDE.md`, and `TESTING_CHECKLIST.md`.

## 1. Mission (Group 2)

This repository is Group 2's track: the quantum-algorithm and search-engine side of the project. Its question, from [`TASK.md`](TASK.md):

> **Can quantum-assisted hypothesis search improve the adaptive reasoning performance of a frontier LLM?**

**Quantum-Assisted Hypothesis Search (QAHS)** targets reasoning in new environments directly. A frontier LLM perceives an unfamiliar ARC-AGI-3 interactive game and describes it in a **factorized rule vocabulary**: six dimensions (Player, Goal, Movement, Topology, Interaction, State) with up to 16 options each. That defines up to 16^6 = 16,777,216 candidate world models, indexed by 24 qubits. **QAHS, not the LLM,** searches that space and selects the action expected to best tell the competing world models apart:

$$a^* = \arg\max_{a \in A_t} \mathrm{IG}(a, \mathcal{H} \mid O_t)$$

The environment executes the action, the LLM interprets the result, and the loop repeats. Everything outside the selection mechanism is held identical between the LLM-only, classical-QAHS, and quantum-QAHS conditions.

## 2. Research question and target architecture

> With identical perception, vocabulary, and observations, does selecting the next experiment by **information-gain search over the full factorized hypothesis space** (exact classical vs. quantum-assisted) improve actions to solution, success, and exploration efficiency compared with letting the **LLM choose on its own**? And what quantum resources does that search need as the space grows from S1 (256) to S5 (16.8M)?

The project does not assume the quantum condition will win. The quantum algorithm must be justified mathematically, not chosen because it sounds promising.

```
ARC state S_t ─► LLM (perceive, fill vocabulary, parse o_t) ─► H, predictions, O_t
                                                                │
                      ARC executes a_t ◄── QAHS selects a_t = argmax IG ◄┘
```

### Target vs. current implementation (as of 2026-10-29)

| Aspect | Target (TASK.md) | Current code |
|---|---|---|
| Hypotheses | Fixed factorized space, 6 × 16 options, 24-bit index | ARC path: 2–8 free-text LLM hypotheses per turn. Controlled path (`factorized.py`): 6 × **2** options |
| Who selects | QAHS only | QAHS selects from LLM-written utilities |
| Objective | IG over full H given O_t | ARC path: hand-tuned per-action utility, so QAHS always picks the top-utility action. Controlled path: exact IG |
| Quantum method | Mathematically justified (QAE / Grover / QAOA / annealing …) | QAOA over a separable cost, not justified |

The ARC harness described in §4 is the **current** implementation. It works end to end but does not yet meet the TASK constraint. The 64-world controlled track is the prototype of the target representation. See [`IMPROVEMENTS.md`](IMPROVEMENTS.md) for the phased plan.

The current harness compares these conditions:

| Condition (`--condition`) | Trace field | Selector |
|---|---|---|
| `llm_only` | `without_qahs` | The LLM's own `recommended_candidate_id` |
| `qahs_classical` | `with_classical` | Exact minimum over the encoded cost vector (`search/classical.py`) |
| `qahs_qaoa` | `with_qahs` | Simulated QAOA, p=2, 3 COBYLA restarts, 80 iterations, 512 shots (`search/qaoa.py`) |

## 3. What the project explicitly does **not** claim

- It is **not** evidence of quantum advantage or quantum speedup.
- QAOA runs as an exact NumPy statevector simulation, not on quantum hardware.
- The full cost diagonal is computed classically; its size grows as `2^qubits`.
- The LLM generates every hypothesis, action, risk, and progress estimate. QAHS can only choose among what it is given.
- Wall time includes model latency and cache effects and must not be read as search speed.
- Demo-environment results are harness checks, not research evidence.

## 4. The reasoning loop (one turn)

Implemented in `src/qahs/arc_agent.py::run_arc_episode`.

1. **Observe** — `ArcSDKEnvironment` (`arc_environment.py`) wraps the official `arc_agi` toolkit and turns the last frame into an `ArcObservation` (grid, state, levels, available actions).
2. **Propose** — `OpenAIArcReasoner.propose` (`arc_reasoner.py`) sends a compact grid description (`frame_summary`), the belief ledger, and recent history. It gets back schema-constrained JSON: 2–8 hypotheses, each predicting an outcome for every one of 2–8 action candidates, plus the LLM's own recommendation. `parse_proposal` validates and clamps the result.
3. **Score** — `score_arc_actions` (`arc_objective.py`) computes, for each candidate:
   `utility = information_gain + 0.65·expected_progress − 0.65·risk + novelty + outcome_bonus − repetition_penalty`.
   Information gain is the expected entropy reduction over hypothesis confidences, with hypotheses grouped by their predicted outcome.
4. **Select** — For QAHS conditions, `ArcEncoding` packs each (hypothesis, action) pair into `ceil(log2 H) + ceil(log2 A)` bits ("decision qubits"). `build_arc_costs` assigns `cost = −utility(action) − 0.05·belief(hypothesis)` and gives invalid encodings a penalty of 8.0. The classical or QAOA backend then picks a low-cost state.
5. **Act** — The chosen action runs in a separate environment instance for that condition.
6. **Reflect** — `OpenAIArcReasoner.reflect` makes a second structured call that compares predictions with what happened and returns a **belief ledger**: each belief is `supported`, `weakened`, `rejected`, `uncertain`, or `new`. The next turn's proposal receives this ledger, so evidence accumulates across turns.

### Fairness mechanisms
- `CachingArcReasoner` gives the same proposal and reflection whenever two conditions reach the same observation with the same history. In practice, turn 1 is shared and the conditions diverge after their first different action.
- `run_arc_comparison` runs the three conditions one after another on fresh environments with the same game and seed.
- `action_signature` (`type:x:y`) identifies the same physical action even when the LLM renames a candidate. It drives the novelty and repetition terms.

## 5. Two experiment tracks

1. **Live ARC-AGI-3 harness** (main track): `qahs-arc` CLI (`arc_cli.py`). Needs `OPENAI_API_KEY`, `OPENAI_MODEL`, and `ARC_API_KEY`. `--source demo` runs a key-free deterministic "hidden door" world (`DemoArcEnvironment` + `DemoArcReasoner`) built to make the conditions visibly differ.
2. **Controlled synthetic track** (validates encoding and search without LLM or API noise):
   - 64-world factorized benchmark: six binary rule dimensions → `2^6 = 64` world models (`factorized.py`, `factorized_objective.py`, `factorized_agent.py`, `qubo.py`, `benchmark.py`, `benchmark_cli.py`, `scaling.py`).
   - Older four-rule teaching demo (`agent.py`, `cli.py`, `environment.py`, `rule_environment.py`, `compiler.py`, `objective.py`, `interpreter.py`, `predictor.py`, `models.py`).

## 6. Data flow and publishing

```
qahs-arc ──► output/arc_runs/*.json        (private, gitignored, full traces)
             output/arc_recordings/        (official toolkit recordings, gitignored)
                    │
scripts/export_dashboard_runs.py  (redacts secret-like keys and error text)
                    ▼
frontend/public/runs/*.json + index.json   (committed) ──► Next.js on Vercel
```

- `frontend/app/page.tsx` is the About page. `frontend/app/results/page.tsx` + `components/Dashboard.tsx` are the results dashboard (charts, grids, turn traces, belief ledger).
- `scripts/local_run_server.py` is a loopback-only HTTP bridge on `127.0.0.1:8765`. It lets the local dashboard start a `compare` run. It is never deployed.
- `run_dashboard.sh` starts the export, the runner, and `next dev` together.

## 7. Evidence so far (as of 2026-10-29)

**Controlled track** (`EXPERIMENT_NOTES.md`): on the 64-world benchmark, exact classical search and the policy proxy both solve 64/64 worlds in about 7 actions. QAOA smoke runs solve 8/8. The environment is intentionally easy and cannot tell the methods apart.

**Live ARC-AGI-3 track** (`frontend/public/runs/`, 9 live comparisons + 2 older-harness live runs + 5 demo runs; games ls20, ft09, ar25, vc33, lp85, r11l, s5i5):

- **No condition won any live game.** The best result was 1 completed level, reached once by LLM-only (r11l, 17:09) and once by classical (r11l, 18:49).
- Turn-level analysis of the saved traces: exact classical selection chose the highest-utility action on **190/190** turns, QAOA on **205/214**, and the LLM on **161/231**. This means the current cost function reduces QAHS to "pick the action with the highest hand-crafted utility." See IMPROVEMENTS A1.
- One s5i5 run (18:49:51) failed in all three conditions before the first action.
- The first seven run files come from an older harness that had no classical condition and no reflection ledger. Do not pool them with newer runs.

**Bottom line:** the software loop works end to end and is well instrumented. There is not yet any evidence that QAHS or QAOA selection improves reasoning.

## 8. Glossary

- **Hypothesis**: an LLM-proposed rule of the hidden game, with a confidence and a predicted outcome per candidate action.
- **Action candidate**: a concrete ARC action (`ACTION1`–`ACTION7`; `ACTION6` is a click at x, y in 0..63) with LLM-estimated progress and risk.
- **Belief ledger**: the set of beliefs carried across turns, with status labels updated by reflection.
- **Decision qubits**: the number of bits needed to encode one (hypothesis, action) pair.
- **Action budget**: the maximum number of actions per condition (`--max-steps`).
- **Rule dimension / register**: one of P, G, M, T, I, S. Each is encoded as a 4-bit register (16 options). Six registers make up the 24-qubit hypothesis index.
- **World model**: one complete hypothesis, i.e. one option per dimension, e.g. (P2, G9, M1, T6, I3, S5).
- **O_t**: all observations collected up to turn t.
- **IG(a, H | O_t)**: the expected reduction in uncertainty over world models from taking action a.
- **S1–S5**: hypothesis-space scales from 256 (8 qubits) to 16,777,216 (24 qubits) used for the scaling study.

## 9. Module map

| Path | Role |
|---|---|
| `src/qahs/arc_agent.py` | Episode loop, three-condition comparison, trace saving |
| `src/qahs/arc_reasoner.py` | Prompts, JSON schemas, parsers, demo reasoner, matched cache |
| `src/qahs/arc_objective.py` | Information gain, utility scoring, cost vector |
| `src/qahs/arc_encoding.py` | Binary encoding of hypothesis–action pairs |
| `src/qahs/arc_environment.py` | ARC toolkit adapter, game ID validation, demo environment |
| `src/qahs/arc_models.py` | Dataclasses for observations, proposals, beliefs, traces |
| `src/qahs/search/` | `ClassicalSearch`, `QAOASearch`, `GreedyBeliefSearch` |
| `src/qahs/providers/openai_provider.py` | OpenAI Responses API with strict JSON schema |
| `src/qahs/arc_cli.py` | `qahs-arc` entry point |
| `src/qahs/factorized*.py`, `qubo.py`, `benchmark*.py`, `scaling.py` | Controlled 64-world track |
| `src/qahs/agent.py`, `cli.py`, `environment.py`, … | Legacy four-rule teaching demo |
| `scripts/` | Export/sanitize runs, local runner service |
| `frontend/` | Next.js About page and results dashboard |
| `environment_files/` | Downloaded ARC game sources and metadata (not used by the code yet) |
| `tests/` | 16 unittest cases (ARC integration + controlled track) |
| `output/pdf/` | Course PDFs: codebase overview and plain-language guide |
