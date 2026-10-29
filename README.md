# Quantum-Assisted Hypothesis Search (QAHS)

QAHS is a research prototype for testing whether quantum-assisted hypothesis selection can strengthen an AI system's reasoning on unfamiliar problems. ARC-AGI-3 provides the test environment: the AI receives an unknown interactive puzzle, proposes possible rules, chooses an action, observes the result, and revises its beliefs.

The project compares three decision methods while holding the game, seed, action budget, language model, prompts, and belief-update process constant:

1. **LLM only** — the language model chooses from its own proposed actions.
2. **Classical QAHS** — exact classical search selects the lowest-cost hypothesis–action pairing.
3. **Quantum QAHS** — a simulated Quantum Approximate Optimization Algorithm (QAOA) searches the same encoded pairings and objective.

The comparison asks whether changing the selection method improves progress, action efficiency, exploration, or computational cost. It does not assume that the quantum condition will win.

## Current reasoning loop

Each condition uses the same six-stage loop:

1. **Observe** the current ARC-AGI-3 grid and game state.
2. **Propose** two to eight competing hypotheses and concrete tests.
3. **Score** actions using information gain, expected progress, risk, novelty, and prior outcomes.
4. **Select** an action using the assigned LLM, classical, or QAOA method.
5. **Act** in a separate instance of the same game and seed.
6. **Reflect** on the result and update a persistent belief ledger.

The post-action reflection is a second structured model call. It compares predicted and observed outcomes, then marks each belief as `supported`, `weakened`, `rejected`, `uncertain`, or `new`. The next turn receives that ledger so evidence is carried forward instead of silently replacing the hypothesis set.

## Scientific boundaries

This is a proof of concept, not evidence of quantum advantage.

- Quantum QAHS currently uses an exact NumPy statevector simulation, not quantum hardware.
- The complete cost diagonal is constructed classically and grows exponentially with the encoded decision space.
- QAOA is an approximate optimizer; exact classical search is expected to find the minimum of the current small search space reliably.
- The LLM creates the hypotheses, actions, and estimated progress. QAHS can select only from the candidates it receives.
- Proposal and reflection responses are cached when two conditions reach identical states with identical histories. This improves experimental consistency, but total wall time must not be interpreted as quantum speedup without separating model latency from search time.
- Conclusions require repeated games and seeds. Failures, interrupted API calls, and action-budget exhaustion must remain in the research record.

Runs created before the persistent reflection ledger was added use an older harness and should not be pooled directly with new runs as though they used the same reasoning process.

## Install

From the repository root:

```bash
bash setup_arc.sh
```

The setup creates `.venv` and installs the local package, OpenAI integration, ARC-AGI toolkit, simulator dependencies, and frontend packages. Activate it afterward with `source .venv/bin/activate` when using the shorter command names.

## Configure API access

Keep credentials in shell environment variables. Never place real keys in `.env.example` or commit them to Git.

```bash
export OPENAI_API_KEY='your-openai-key'
export OPENAI_MODEL='a-structured-output-capable-model'
export ARC_API_KEY='your-arc-api-key'
```

`.gitignore` excludes `.env`, the virtual environment, raw run JSON, recordings, frontend build output, and installed Node packages.

## Run an experiment

First verify the complete comparison locally without API keys:

```bash
.venv/bin/qahs-arc \
  --source demo \
  --condition compare \
  --max-steps 6
```

Run a live ARC-AGI-3 comparison:

```bash
.venv/bin/qahs-arc \
  --source arc \
  --game r11l \
  --condition compare \
  --seed 0 \
  --max-steps 20
```

Supported conditions are:

- `compare` — run all three matched conditions.
- `llm_only` — run only the language-model recommendation.
- `qahs_classical` — run exact classical QAHS.
- `qahs_qaoa` — run simulated quantum QAHS.

An uncached comparison can make up to `conditions × actions × 2` model calls because every action may require one proposal and one reflection. For example, three 20-action conditions can require up to 120 calls. Shared cache hits can reduce that number.

Raw working traces are saved under `output/arc_runs/` and are intentionally ignored by Git.

## Run the website

```bash
bash run_dashboard.sh
```

The launcher:

1. exports sanitized copies of saved experiments;
2. starts the loopback-only Python experiment service;
3. starts the Next.js application; and
4. cleans up its local processes when stopped with `Ctrl+C`.

Open `http://localhost:3000`:

- `/` — plain-language About page, research hypothesis, learning-loop infographic, and current architecture diagram.
- `/results` — saved-run comparison dashboard and detailed turn traces.
- The local Results page includes a run panel. It is hidden when the static site is hosted publicly.

New traces include the post-action reflection inside each turn's details. Older traces remain readable but do not contain this field.

## Publish sanitized results

Refresh the public dashboard data after local experiments:

```bash
.venv/bin/python scripts/export_dashboard_runs.py
git add frontend/public/runs
git commit -m "Add latest ARC-AGI-3 experiment runs"
git push origin main
```

Only the sanitized files under `frontend/public/runs/` are published. API keys, raw traces, and local recordings are not sent to the frontend. A connected Vercel project rebuilds automatically after a push to `main`.

Vercel should import `frontend` as a single Next.js project. The Python experiment service remains local and is not deployed.

## Controlled 64-hypothesis experiment

The repository also retains a smaller synthetic experiment with six binary rule dimensions and `2^6 = 64` possible world models. It is useful for validating encoding, objectives, search behavior, and scaling independently of model or ARC API reliability.

Run exact conditions across all 64 worlds:

```bash
PYTHONPATH=src python -m qahs.benchmark_cli \
  --conditions policy_proxy classical \
  --worlds 64
```

Run a small QAOA subset first because statevector simulation is exponential:

```bash
PYTHONPATH=src python -m qahs.benchmark_cli \
  --conditions qaoa \
  --worlds 4 \
  --output output/experiments/qaoa_smoke.json
```

## Testing

```bash
.venv/bin/python -m unittest discover -s tests -v
cd frontend
npm run build
```

The tests cover game-ID validation, structured response parsing, persistent reflection records, action scoring, classical search, QAOA search, encoding, and the local three-condition demonstration.

## Repository map

### ARC experiment path

- `src/qahs/arc_environment.py` — ARC-AGI-3 toolkit adapter and deterministic test environment.
- `src/qahs/arc_reasoner.py` — structured proposal prompt, post-action reflection prompt, persistent belief ledger, parsers, and matched-response cache.
- `src/qahs/arc_objective.py` — information gain, progress, novelty, risk, and repetition scoring.
- `src/qahs/arc_encoding.py` — binary encoding of hypothesis–action pairs.
- `src/qahs/arc_agent.py` — observe–propose–select–act–reflect loop and matched comparison.
- `src/qahs/search/` — exact classical and simulated QAOA selectors.
- `src/qahs/arc_cli.py` — `qahs-arc` command-line entry point.
- `src/qahs/providers/` — OpenAI structured-output adapter.

### Website and storage

- `frontend/app/page.tsx` — About homepage and web-native project diagrams.
- `frontend/app/results/page.tsx` — experiment dashboard route.
- `frontend/components/Dashboard.tsx` — charts, run selection, trace details, and local run panel.
- `scripts/local_run_server.py` — loopback-only bridge from the local website to `qahs-arc`.
- `scripts/export_dashboard_runs.py` — sanitizes private traces for the public frontend.
- `frontend/public/runs/` — display-only experiment files intended for Git and Vercel.
- `run_dashboard.sh` — local website and runner launcher.

### Controlled research components

- `src/qahs/factorized.py` — six-dimension hypothesis representation.
- `src/qahs/qubo.py` — QUBO and Ising conversion.
- `src/qahs/factorized_agent.py` — controlled hidden-world loop.
- `src/qahs/benchmark.py` — repeated conditions and aggregate metrics.
- `src/qahs/scaling.py` — qubit and simulator-memory estimates.
- `tests/` — Python correctness and integration tests.

Additional operating instructions are available in `ARC_RUN_GUIDE.md` and `TESTING_CHECKLIST.md`.
