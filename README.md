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
2. **Propose** eight to sixteen structured hypotheses over a deterministic catalog of up to 32 two-step plans.
3. **Score** plans using information gain, expected progress, survival, risk, diversity, and prior outcomes.
4. **Select** a plan using the assigned LLM, classical, or QAOA method, execute its first step, and replan from the new observation.
5. **Act** in a separate instance of the same game and seed.
6. **Reflect** on the result and update a persistent belief ledger.

The v6 harness separates mechanism confidence from goal relevance, validates model-supplied hypothesis roles, groups semantic duplicates, increases alternative-goal pressure after stagnation, and records every objective contribution. It uses an adaptive 16-plan catalog, confidence saturation, goal–action alignment, progress credit, one-shot policy transfer between levels, periodic full-grid refreshes with compact deltas, validated second actions, and event-triggered model review. Saved traces retain the complete evidence and diagnostics.

## Scientific boundaries

This is a proof of concept, not evidence of quantum advantage.

- Quantum QAHS currently uses an exact NumPy statevector simulation, not quantum hardware.
- The complete cost diagonal is constructed classically and grows exponentially with the encoded decision space.
- QAOA is an approximate optimizer; exact classical search is expected to find the minimum of the current small search space reliably.
- The harness generates candidate actions and two-step plans from available controls, visible objects, changed cells, boundaries, and unexplored regions. The LLM supplies structured hypotheses, predictions, progress, and risk estimates.
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
  --batch batch-01 \
  --seed 0 \
  --max-steps 20
```

Supported conditions are:

- `compare` — run all three matched conditions.
- `llm_only` — run only the language-model recommendation.
- `qahs_classical` — run exact classical QAHS.
- `qahs_qaoa` — run simulated quantum QAHS.

V6 can handle predictable routine outcomes locally and invoke the model on surprises, interactions, progress, major changes, strategy resets, and periodic verification. Consequently, call counts depend on the game rather than being fixed at approximately one request per action.

Raw working traces are saved under `output/arc_runs/` and are intentionally ignored by Git.

New traces use a self-describing filename and store the same fields inside the JSON:

```text
20261030T190000Z__v5-compact-combined__batch-01__r11l__seed-00__actions-20__gpt-6-luna__comparison.json
```

The harness version distinguishes reflection-enabled experiments from older runs. Use one batch ID for runs that intentionally share a harness, model, seed, action budget, and game set. If a game is repeated within a batch, the dashboard uses its newest run.

### Using Claude instead of OpenAI

`--provider anthropic` swaps only the model behind every harness (v7, v10, v11, v12); prompts, schemas, images, scoring, and selectors are unchanged.

```bash
.venv/bin/pip install -e '.[anthropic]'
export ANTHROPIC_API_KEY='your-anthropic-key'
export ANTHROPIC_MODEL='claude-opus-5-5'   # default
export ANTHROPIC_EFFORT='medium'           # default; low, medium, high, xhigh, or max
.venv/bin/qahs-arc --source arc --provider anthropic --harness v12 --game r11l --condition compare --seed 0 --max-steps 20
```

Anthropic structured outputs cannot enforce `maxItems`, `minItems` above 1, or numeric and string bounds, which OpenAI strict mode enforces while decoding. The Claude caller restates those constraints in the schema descriptions, retries a response that breaks them up to twice, and then cuts over-long arrays to `maxItems`. The retry and truncation counts are recorded in the run's `model_settings`. Server-side model fallbacks are off, so every trace comes from a single model. The local dashboard still launches OpenAI runs only.

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
- Results can switch between a single run and a versioned batch. Batch view supports metric and mode filters across games and warns when experimental settings do not match.
- The local Results page includes a run panel. Choose **Single** for one game or **Batch** to enter several game IDs with one shared batch ID, seed, and action budget. Batch jobs run sequentially and report progress in the panel. The run panel is hidden when the static site is hosted publicly.

New traces include structured reflections, representation metadata, per-stage timing, and QAOA regret/rank diagnostics inside each turn. Older traces remain readable without these fields.

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

# V11 information-gain search

ARC-AGI-3 runs now default to `v11-information-gain-search`. Luna proposes competing complete hypotheses and predicts the discrete outcome of every available physical test. Python classifies the measured outcome and updates hypothesis probabilities; exact classical search and QAOA then optimize the same ordered three-test information-gain objective, execute only the first action, and replan. Traces retain the prediction matrix, probability changes, plan, and measured outcome.

V10 is frozen as `v10-success-conditioned-revision` and remains available with `qahs-arc --harness v10 ...` or the local dashboard harness selector. V11 lives in a separate module so V10 behavior and existing results are not silently changed.
