# Quantum-Assisted Hypothesis Search (QAHS)

This is a separate proof-of-concept for the updated Group 2 assignment. It does not depend on `SuperAI-QML`.

The first environment is a tiny hidden-rule game. A blue player sees a red tile and a green door. The interpreter proposes four explanations for the red tile. The objective measures which action would provide the most information, then either exact classical search or QAOA selects an encoded hypothesis-action combination. The environment executes the action, the beliefs are updated, and the loop repeats.

## Run the command-line demonstration

```bash
python -m qahs.cli --backend classical
python -m qahs.cli --backend qaoa
```

If the package has not been installed, run from the project root with:

```bash
PYTHONPATH=src python -m qahs.cli --backend qaoa
```

Run traces are written to `output/runs/` as JSON.

## Current scientific boundary

The included interpreter is deterministic so the search loop is testable without an API key. `StructuredLLMInterpreter` accepts any model-calling function that returns the documented JSON schema. The QAOA implementation adapts the statevector, cost-phase, mixer, COBYLA, and sampling pattern from the instructor-provided malware clustering notebook. Its malware graph and Max-Cut objective are not reused; QAHS supplies its own information-gain objective.

This small implementation validates the software contract. It does not establish a quantum advantage. The current prototype constructs the complete cost diagonal classically and uses exact statevector simulation; both operations scale exponentially. A research version must replace that enumerated table with a compact, justified cost Hamiltonian or oracle that calculates utility from encoded hypothesis fields. The next experimental step is to add an ARC adapter and compare identical hypothesis/action inputs across the greedy, exact-classical, and QAOA backends.

## 64-hypothesis experiment

The second experiment implements six binary rule dimensions: Player, Goal, Movement, Topology, Interaction, and State. Their Cartesian product contains `2^6 = 64` complete world models. It uses a compact QUBO with six hypothesis bits and seven one-hot action variables, for 13 decision qubits in the current formulation.

Run the two inexpensive matched conditions across all 64 hidden worlds:

```bash
PYTHONPATH=src python -m qahs.benchmark_cli --conditions policy_proxy classical --worlds 64
```

Run QAOA on a small subset first because exact statevector simulation is exponential:

```bash
PYTHONPATH=src python -m qahs.benchmark_cli --conditions qaoa --worlds 4 --output output/experiments/qaoa_smoke.json
```

The generated JSON records solve rate, actions to solution, wall time, shots, optimizer evaluations, and qubit estimates. The QUBO is compact, but the simulator still expands it into a complete cost diagonal. That expansion is acceptable for correctness testing and must not be presented as a scalable quantum advantage.

## Optional OpenAI interpreter

The deterministic interpreter remains the default. To enable the API adapter later:

```bash
python -m pip install -e '.[llm]'
export OPENAI_API_KEY='your-key'
export OPENAI_MODEL='a-model-available-to-your-project'
```

`qahs.providers.OpenAIJSONCaller` reads credentials only from the environment and can be passed to `StructuredLLMInterpreter`. Real keys are excluded from source files and traces.

## ARC-AGI-3 experiments and dashboard

Install the complete local environment once:

```bash
python3 -m venv .venv
source .venv/bin/activate
bash setup_arc.sh
```

The same setup is available as `bash setup_arc.sh`. After setup, `bash run_dashboard.sh` exports sanitized saved runs and launches the Next.js dashboard.
See `ARC_RUN_GUIDE.md` for the short first-run sequence and expected output.

Verify the comparison without API keys:

```bash
qahs-arc --source demo --condition compare --max-steps 6
bash run_dashboard.sh
```

The dashboard opens at `http://localhost:3000`. On localhost it includes a Run panel that invokes the Python engine; the panel is hidden in the static Vercel build. The demonstration runs the same hidden-door world three times: the control follows the reasoner's recommendation, exact classical QAHS optimizes the scored candidates deterministically, and quantum QAHS uses simulated QAOA on the same objective.

For a real ARC-AGI-3 run, place credentials in the shell rather than source code:

```bash
export OPENAI_API_KEY='your-openai-key'
export OPENAI_MODEL='a-structured-output-capable-model-in-your-project'
export ARC_API_KEY='your-arc-key'

qahs-arc --source arc --game ls20 --condition compare --max-steps 12
.venv/bin/python scripts/export_dashboard_runs.py
bash run_dashboard.sh
```

A 12-step comparison can make up to 36 model calls because each condition receives its own game instance. The Python command saves private working traces under `output/arc_runs/`; the exporter writes sanitized, display-only copies under `frontend/public/runs/`. The Next.js dashboard never receives API credentials and cannot launch experiments on Vercel.

The official toolkit documents anonymous access, but the live service returned HTTP 401 during verification on October 29, 2026. The project therefore requires `ARC_API_KEY` for live runs and keeps the local demonstration available for key-free testing.

### What is being compared

- **Without QAHS:** the language model proposes hypotheses and actions, then its own `recommended_candidate_id` is executed.
- **Classical QAHS:** exact classical search selects from the QAHS-scored candidates.
- **Quantum QAHS:** QAOA selects from the same candidates and objective.
- All runs use the same game ID, seed, action budget, prompt structure, and model configuration. They are separate game sessions because their selected actions can produce different future states.
- Identical state/history inputs are cached across conditions, so the shared initial state uses the same model proposal rather than two potentially different API responses.
- Hypothesis slots persist across turns: the model retains plausible explanations, updates their confidence after observations, and replaces a slot only when evidence contradicts it.
- Exact classical QAHS and QAOA use the same candidates, scores, and cost function, isolating the effect of the selector.

This is an experimental harness, not evidence of quantum advantage. The current QAOA implementation is an exact NumPy statevector simulation. Real comparisons should use multiple games and seeds, retain failures, and report action efficiency as well as completion.

## Layout

- `src/qahs/models.py`: shared data structures.
- `src/qahs/environment.py`: hidden-rule game.
- `src/qahs/interpreter.py`: deterministic and API-ready interpreters.
- `src/qahs/compiler.py`: binary encoding.
- `src/qahs/predictor.py`: predicted outcomes under each hypothesis.
- `src/qahs/objective.py`: information gain, belief update, and cost diagonal.
- `src/qahs/factorized.py`: six-dimension, 64-hypothesis representation.
- `src/qahs/rule_environment.py`: controlled hidden-world experiment.
- `src/qahs/arc_environment.py`: official ARC-AGI-3 toolkit adapter plus a local test environment.
- `src/qahs/arc_reasoner.py`: structured OpenAI hypothesis/action proposals and deterministic test reasoner.
- `src/qahs/arc_agent.py`: matched with/without-QAHS experiment loop and trace storage.
- `frontend/`: Next.js comparison dashboard and Vercel-ready static export.
- `scripts/export_dashboard_runs.py`: sanitizes saved traces for the frontend.
- `src/qahs/qubo.py`: compact QUBO and Ising conversion.
- `src/qahs/factorized_agent.py`: matched 64-hypothesis loop.
- `src/qahs/benchmark.py`: repeated conditions and aggregate metrics.
- `src/qahs/scaling.py`: qubit and simulator-memory estimates.
- `src/qahs/providers/`: optional OpenAI Responses API adapter.
- `src/qahs/search/`: greedy, exact classical, and QAOA backends.
- `src/qahs/agent.py`: complete observe-search-act-update harness.
- `src/qahs/run_store.py`: JSON trace storage for the teaching CLI.
- `tests/`: correctness tests for the loop and both search paths.

