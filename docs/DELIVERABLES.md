# Group 2 Deliverables — Answer Sheet

Short answers to each deliverable in [`TASK.md`](TASK.md). **Answer** gives the design or justification; **Status** says what exists today. Updated 2026-10-29.

---

## 1. Formal finite hypothesis representation (compatible with LLM output)

**Answer**
- H = H_P × H_G × H_M × H_T × H_I × H_S: 6 registers × 4 bits = a **24-bit index**.
- Register order P, G, M, T, I, S (most significant first): `x = 16⁵·p + 16⁴·g + 16³·m + 16²·t + 16·i + s`.
- Each option = (index, English description, categorical prediction per action).
- The LLM returns **JSON options per dimension** grounded in the frame (colors, objects), not free-text hypotheses.
- Example: (P0, G0, M0, T0, I1, S0) → `0000 0000 0000 0000 0001 0000`.
- Invalid states (a dimension with k_d < 16 options): masked out classically; for quantum, zero amplitude at state preparation, or an oracle penalty / phase flag.
- Ancillas depend on the algorithm (see §5).

**Status**
- A 2-options-per-dimension version exists (`src/qahs/factorized.py`, 64 worlds). The 16-option version is still to build.

## 2. Mathematically justified search / utility objective

**Answer**
- Posterior: `P(h | O_t) ∝ prior(h) · Π_k 1[f(h, a_k) = o_k]`. The prior can include a complexity penalty `exp(−β·complexity(h))`.
- Predicted outcome distribution: `p(o | a) = Σ_h P(h | O_t) · 1[f(h, a) = o]`.
- With deterministic predictions: **`IG(a) = H(O | a) = −Σ_o p(o | a) · log p(o | a)`**. The best action is the one that splits the posterior mass most evenly.
- Full score: `U(a) = IG(a) + λ_g·P(progress | a) − λ_c·cost(a)`. Each term can be switched on or off for ablations.
- This answers TASK.md's central question directly. "Follow the top hypothesis" only exploits what the agent already believes; IG chooses the action that best discriminates.

**Status**
- Exact IG exists for 64 worlds (`src/qahs/factorized_objective.py`).
- The ARC path still uses a heuristic utility: classical picked the top-utility action on 190/190 turns.

## 3. Working QAHS implementation (Python/Jupyter, simulator)

**Answer**
- API: `select_action(space, predictions, observations, actions, backend) → (action, score, diagnostics)`.
- Backends: `classical_exact`, `qae`, `grover`, `qaoa`.
- Loop: ARC → LLM (perceive, fill in the vocabulary, parse o_t) → QAHS → action → ARC.
- Stops on an environment-verified WIN or when the action budget runs out. The LLM's own solved / not-solved declaration is logged.
- Simulators: Qiskit Aer plus the existing NumPy statevector.
- Notebooks: `01_representation`, `02_objective`, `03_quantum_formulations`, `04_scaling`, `05_arc_experiment`.

**Status**
- Exists: the ARC harness, NumPy QAOA (`search/qaoa.py`), QUBO/Ising conversion (`qubo.py`), and the Phase 0 robustness work.
- Not yet built: the 16⁶ engine, the QAE and Grover backends, and the notebooks.

## 4. Justification for the selected quantum algorithm

**Answer**
- There are only 4–7 actions, so taking the argmax is trivial. The expensive part is summing over 16.7M hypotheses. That makes this an **estimation** problem, not an optimization problem.
- **QAE (primary):** estimates p(o | a) to error ε in O(1/ε) oracle calls, versus O(1/ε²) classical samples. It plugs directly into IG = H(O | a).
- **Grover:** samples hypotheses consistent with O_t in O(√(N/M)) calls instead of O(N/M) (N hypotheses, M of them consistent).
- **QAOA / QUBO / annealing:** only for combinatorial choices, such as d-step plans (|A|^d options) or picking the single most likely hypothesis. The current QAOA runs over a separable cost, which isn't justified, so it is kept only as a comparison.
- **Caveat:** classical exact enumeration of 16⁶ takes seconds (about 67 MB as float32). Any advantage is a query-complexity and scaling claim, and state-preparation and oracle costs can cancel it.

**Status**
- QAOA is implemented. QAE and Grover are not yet.

## 5. Resource and scaling analysis

**Answer**
- Index qubits for S1–S5: 8, ≈13, 18, ≈22, 24.
- On top of that: action register (2–3 qubits), outcome ancilla(s), QAE precision register m (6–8), and oracle workspace. Estimated **35–45 logical qubits at S5** (to be measured).
- Statevector memory is 2ⁿ × 16 B: 24 qubits ≈ 268 MB, 30 qubits ≈ 17 GB. So full QAE at S5 can't be simulated as a statevector.
- Workarounds: oracles limited to the dimensions an action depends on (4k qubits for k dimensions), or an MPS simulator.
- Measure: transpiled depth, two-qubit gates, shots, oracle calls, iterations, wall time, classical compute, and noisy vs. noiseless results.
- Sweeps: N = 8 … 256, then S1 → S5; look-ahead depth d = 1–3.

**Status**
- `src/qahs/scaling.py` only estimates statevector memory.

## 6. Correctness tests at small scale, then scaling toward ~16.8M

**Answer**
- Representation: encode/decode round-trip at every scale, the worked example, invalid-index masking.
- Objective: IG checked by hand at S1; IG = 0 when every hypothesis predicts the same outcome; IG = log₂ k for an even k-way split.
- Quantum vs. classical:
  - QAE estimates within ε of the exact p(o | a);
  - Grover returns only consistent hypotheses;
  - QAOA approximation ratio and probability of the optimum.
- Scale up only after S1 and S2 match the exact classical results.

**Status**
- 41 tests pass: encoding, QUBO = Ising energies, QAOA, the 64-world episode, Phase 0 robustness, grid encodings, and session recovery.

## 7. Experimental comparison: LLM-only vs QAHS with identical inputs

**Answer**
- Conditions: LLM-only, LLM + QAHS (classical exact), LLM + QAHS (quantum), plus random and greedy baselines.
- Held identical: game, seed, action budget, model, prompts, vocabulary, and observations. A shared response cache, with the condition order rotated across seeds.
- Games: ar25 plus the 6 other local games in `environment_files/`, at least 5 seeds each. Offline mode during development.
- Primary metrics and the success definition come from Group 1's pre-registration, frozen before held-out runs.

**Status**
- The three-condition harness, shared cache, and rotated order exist. The conditions don't yet use the factorized representation.

## 8. Experimental results: LLM-only vs LLM + QAHS

**Answer (results so far)**
- Live ARC: 11 runs over 7 games, 9 of them with all three conditions. **No condition won any game.** The best result was 1 level, reached once by LLM-only and once by classical.
- These runs **don't test QAHS yet**: classical matched the top-utility action on 190/190 turns and QAOA on 205/214.
- Controlled 64-world track: classical and the policy proxy each solved 64/64 worlds in about 7 actions; QAOA solved 8/8 smoke worlds. The environment is too easy to separate the methods.
- **No evidence for or against QAHS yet.** Final results depend on the roadmap phases 1–6 ([`IMPROVEMENTS.md`](IMPROVEMENTS.md)).
- Even a significant win would only show "Stronger AI" characteristics under Group 1's definition. It is not proof of a quantum advantage.
