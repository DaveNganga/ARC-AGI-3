# QAHS — Roadmap to the Group 2 Deliverables

Updated 2026-10-29 after receiving the Group 2 assignment ([`TASK.md`](TASK.md)). Earlier findings came from reading the code and analyzing the 16 saved runs in `frontend/public/runs/`. See [`CORE.md`](CORE.md) for project context.

## 1. Summary

The code has a working, well-instrumented observe → propose → select → act → reflect loop on live ARC-AGI-3 games. It does **not** yet implement what TASK.md asks for:

1. **Representation.** TASK.md requires a fixed factorized space P×G×M×T×I×S with 16 options per dimension (16^6 world models, 24 qubits). The ARC harness instead uses 2–8 free-text hypotheses invented each turn.
2. **Who decides.** TASK.md says QAHS, not the LLM, searches H and selects the action. Today the LLM writes the hypotheses, each hypothesis's predictions, and the progress and risk scores that drive the choice.
3. **Objective.** TASK.md asks for argmax over IG(a, H | O_t) across the full space. The current cost is a hand-tuned per-action utility with a 0.05 hypothesis tie-break (`arc_objective.py:104`). On saved runs, classical selection picked the top-utility action on **190/190** turns and QAOA on 205/214, so the hypothesis register has no effect.
4. **Algorithm justification.** QAOA was chosen without a mathematical justification, over a cost function that has no hypothesis–action interaction.

The controlled 64-world track already has the right structure: the same six dimensions (`factorized.py`), exact IG with consistency-filter updates (`factorized_objective.py`), and QUBO/Ising conversion (`qubo.py`). `scaling.py::default_scaling_table` already lists the S1–S5 sizes. **That track is the base to build on; the free-text ARC path should become the LLM-only control and the perception front end.**

## 2. Gap analysis

| TASK requirement | Current state | Gap |
|---|---|---|
| Factorized H with 6 dims × 16 options, 24-bit index | `factorized.py`: 6 dims × **2** options (64 worlds) | Generalize to k_d ≤ 16 options per dimension, with fixed 4-bit registers |
| Exact binary encoding, invalid-state handling, ancillas documented | `qubo.OneHotEncoding`: 6 bits + one-hot actions; no padding rules | Specify bit order, padding mask for k_d < 16, and ancilla budget per algorithm |
| LLM instantiates the rule vocabulary from the grid | Vocabulary hardcoded in `RULE_OPTIONS`; ARC LLM writes free text | New LLM schema: per-dimension options grounded in the observed frame |
| LLM must **not** search H or select the action (QAHS condition) | LLM recommendation used for `llm_only`; QAHS uses LLM-written utilities | QAHS selection must depend only on H, predictions, and O_t |
| f(x_i, a_j, O_t): outcome predicted by each world model | ARC: LLM writes free-text predictions per hypothesis. Factorized: each probe reveals one bit | Compositional categorical predictor over real ARC actions |
| a* = argmax IG(a, H \| O_t) over full H | ARC: utility heuristic. Factorized: exact IG over 64 worlds | Exact IG over up to 16^6 as the classical baseline |
| Quantum formulation justified mathematically | QAOA reused from a course notebook | Written justification plus a comparison of candidate formulations |
| Python/Jupyter, externally callable module | Python package + CLI; no notebooks | `notebooks/` and a clean `qahs.search_engine` API |
| Resource and scaling analysis S1–S5, N sweep, depth d | `scaling.py` estimates statevector memory only | Measured resources, sweeps over N and d, noise |
| LLM-only vs LLM+QAHS with identical inputs | 3-condition harness exists, but inputs differ in meaning | Re-wire the harness to the new representation; multiple seeds |
| Log the LLM's solved / not-solved declaration | Not recorded | Add to the schema and the trace |
| Success defined by Group 1 pre-registration | Not referenced | Get the definition from Group 1 and freeze it |

## 3. Key design decisions (proposed)

- **Classical exact IG is feasible at S5.** One float32 belief vector over 16^6 is about 67 MB. A vectorized NumPy pass per action takes seconds. When an action's outcome depends on only a few dimensions, marginalizing makes this far cheaper. **So the classical baseline stays strong at full scale.** Quantum claims must be framed in terms of query complexity and resource scaling, not as "classical can't do it."
- **Match the algorithm to the structure of the problem:**
  - IG(a) depends on the outcome masses p(o | a) = Σ_h P(h | O_t)·[f(h, a) = o]. That is an **expectation over H**, not an optimization, and with |A| = 4–7 the argmax over actions is trivial. → **Quantum amplitude estimation** of p(o | a) is the natural primary formulation: it needs O(1/ε) oracle queries versus O(1/ε²) for classical Monte Carlo.
  - Restricting to hypotheses consistent with O_t is a marked-state search. → **Grover / amplitude amplification** to sample consistent world models.
  - **QAOA / QUBO / annealing** fit only where the decision space is combinatorial: d-step action sequences (|A|^d) or MAP world-model selection. The current QAOA over (hypothesis, action) pairs with a separable cost should be dropped or reframed.
- **Predictor.** The LLM supplies, for each dimension option and each action, a **categorical** outcome (e.g. `no_move | move | blocked | collect | hazard | level_up | toggle | unknown`). Each action declares which dimensions it depends on. f(x, a, O_t) composes those options' predictions. This avoids the free-text outcome problem (old finding A2) and makes the oracle reversible and computable.

## 4. Phased plan

### Phase 0 — Engineering fixes ✅ done 2026-10-29
- [x] **0.1 · P0** Export no longer wipes published runs. `scripts/export_dashboard_runs.py` now merges into `frontend/public/runs/` and does nothing when `output/arc_runs/` is empty. Every `error*` field is redacted.
- [x] **0.2 · P1** `run_dashboard.sh` falls back to a plain background start when `setsid` is missing (macOS) and still cleans up Next.js workers.
- [x] **0.3 · P1** Private traces keep the full error and traceback (`arc_agent.py`). Every episode records `stop_reason`. The local runner prints stderr to the terminal and allows only one run at a time (HTTP 409).
- [x] **0.4 · P1** Model requests are retried with backoff, request and parse together (`src/qahs/retry.py`, `OpenAIArcReasoner._request`). Environment steps are deliberately not retried.
- [x] **0.5 · P1** Condition order rotates by `seed % 3` and is stored as `condition_order`. Optional `OPENAI_TEMPERATURE` is recorded in `model_settings`.
- [x] **0.6 · P2** `GreedyBeliefSearch` works with string actions. Stale docs are fixed. Traces carry `harness_version`, and the dashboard flags legacy runs (no classical condition) and shows `stop_reason`.
- Tests: `tests/test_phase0.py` (10 new tests; 26 in total).
- Old findings A1 (no coupling) and A2 (free-text outcomes) are **superseded by Phases 1–2**.

### Phase 1 — Formal hypothesis representation  → *Deliverable: formal finite representation*
- New `src/qahs/hypothesis_space.py`, generalizing `factorized.py`:
  - `RuleVocabulary`: 6 `Dimension` registers, each with k_d ≤ 16 named options. Each option has an English description and a machine-checkable meaning.
  - Encoding: fixed 4-bit registers in the order P, G, M, T, I, S (MSB → LSB), giving 24 bits. Indices ≥ k_d are **invalid** and get a validity mask (classical) or an oracle penalty / phase flag (quantum). Document the ancilla counts per algorithm.
  - `encode(h) → int`, `decode(int) → tuple`, `natural_language(h)`, `valid_mask()`.
- Scale presets S1–S5, with per-dimension sizes confirmed against TASK.md open question 2 (S1 = 256 and S2 = 7,776 are not k^6).
- Tests: round-trip encode/decode at every scale, the worked example `(P0,G0,M0,T0,I1,S0) → 0000 0000 0000 0000 0001 0000`, and invalid-index handling.

### Phase 2 — Search objective  → *Deliverable: mathematically justified objective*
- New `src/qahs/world_model.py`: the predictor f(x, a, O_t) built from per-dimension categorical predictions, plus the action → relevant-dimensions map.
- Posterior P(h | O_t): uniform or complexity prior × consistency filter. Reuse the pattern in `factorized_objective.update_factorized_beliefs`, vectorized over the encoded space.
- Exact IG(a, H | O_t) via enumeration (S1–S3) and factorized marginalization (S4–S5). This is the **classical baseline**.
- Optional terms, each switchable for ablations: expected progress toward inferred goals, prediction disagreement, complexity penalty, exploration cost.
- Write-up: derivation of IG, why argmax IG answers TASK.md's central question, and how each term is justified.
- Tests: hand-computed IG on S1, and a check that IG is zero when every hypothesis predicts the same outcome.

### Phase 3 — LLM role change (perception and vocabulary only)
- New proposal schema in `arc_reasoner.py`. The LLM gives:
  - the instantiated options per dimension, grounded in the frame (colors, objects);
  - categorical predictions per (option, action);
  - a parsed observation o_t after each action;
  - a solved / not-solved declaration.
- QAHS conditions ignore any LLM action recommendation. LLM-only keeps it, with **identical inputs**.
- Reuse `CachingArcReasoner` for matched inputs, and follow the existing `parse_proposal` / `parse_reflection` validation patterns.
- Prompt states the complete factorized specification explicitly, as TASK.md §1 requires.

### Phase 4 — Quantum formulations  → *Deliverables: QAHS implementation + algorithm justification*
- `src/qahs/quantum/` on a Qiskit simulator, next to the existing NumPy statevector (`search/qaoa.py`):
  - **(a) Amplitude estimation (primary).** A state-preparation oracle for P(h | O_t) and an outcome oracle [f(h, a) = o] estimate p(o | a), which gives IG(a). Report the error ε against oracle calls.
  - **(b) Grover sampling** of consistent world models (oracle = consistency with O_t), with the iteration count derived from the estimated consistent fraction.
  - **(c) QAOA / QUBO** over d-step action sequences, for the look-ahead study. Reuse `qubo.QUBOModel` (QUBO → Ising) and `search/qaoa.py`.
- Externally callable API: `select_action(space, predictions, observations, actions, backend=...) → (action, score, diagnostics)`.
- Written justification comparing (a), (b), and (c) against the problem structure (see §3).

### Phase 5 — Quantum-specific evaluation  → *Deliverables: resource and scaling analysis, correctness tests*
- Correctness first: on S1 and S2, quantum outputs must match exact classical IG and selection within tolerance.
- Sweeps: N = 8, 16, 32, …, 256 and then S1→S5; look-ahead depth d = 1, 2, 3.
- Metrics per TASK.md §5: objective quality (regret vs. exact), hypotheses or experiments evaluated, qubits, ancillas, transpiled depth, two-qubit gates, shots, oracle calls, iterations, wall time, and classical compute.
- Noisy simulator vs. noiseless, and hardware only on a frozen small instance.
- Extend `scaling.py` (measured resources, not just memory estimates) and `benchmark.py` (sweep runner + CSV output).

### Phase 6 — Integrated experiment  → *Deliverables: LLM-only vs LLM+QAHS comparison and results*
- Conditions: **LLM-only**, **LLM + QAHS (classical exact)**, **LLM + QAHS (quantum)**, plus random and greedy baselines.
- Games: ar25 (reference task) and the other downloaded games in `environment_files/`. Use several seeds per game.
- Use Group 1's pre-registered success definition. Freeze prompts, schema, vocabulary, and quantum settings before the held-out runs.
- Report actions to solution, levels, success rate with confidence intervals, and resource costs. Keep failures and budget exhaustion in the record.

### Notebooks (Jupyter deliverable)
`notebooks/01_representation.ipynb` · `02_objective.ipynb` · `03_quantum_formulations.ipynb` · `04_scaling.ipynb` · `05_arc_experiment.ipynb`. Each one imports the package and contains no duplicated logic.

## 5. Ongoing engineering improvements
- CI: `python -m unittest`, `npm run build`, ruff, and a type checker.
- Batch runner over games × seeds × conditions with an aggregate report.
- Use `environment_files/` (downloaded game sources) for offline play, so development doesn't spend API budget.
- Structured JSON history in prompts instead of Python `repr` dumps (`arc_reasoner.py:175-178`).
- Provider interface (OpenAI today; add Claude) so the "same frontier LLM" stays a controlled variable.
- Dashboard: show QAHS IG scores, posterior mass, and quantum resources per turn; mark legacy runs.
- Move `tmp/pdfs/` scripts to `tools/`, and move the four-rule teaching demo to `src/qahs/legacy/`.

## 6. Suggested order
~~Phase 0~~ (done) → Phase 1 → Phase 2 (classical baseline working end to end on S1–S3) → Phase 3 → Phase 4 → Phase 5 → Phase 6. Before Phase 1, resolve the open questions in TASK.md (scale sizes, H2/H3 tuples, Group 1 definitions).

Short answers per deliverable: [`DELIVERABLES.md`](DELIVERABLES.md). Designed but not yet scheduled (grid token encoding, session resume): [`BACKLOG.md`](BACKLOG.md).
