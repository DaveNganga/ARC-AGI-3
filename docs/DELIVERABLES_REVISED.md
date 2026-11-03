# Group 2 Deliverables — Revised Evidence-Based Answer Sheet

This document answers the eight deliverables in `TASK.md` while separating:

1. the **target research design**;
2. what the **current code actually implements**; and
3. the **evidence still required** before a deliverable can be called complete.

It is intentionally conservative. A proposed design is not reported as a finished result, and the existing free-text ARC harness is not treated as equivalent to the target 16.8-million-state factorized QAHS system.

## Branch-state note

This snapshot was written while two lines of work had not yet been reconciled:

- Remote `main` (`7f7a5e7`) contains Phase-0 robustness work: model-call retries, stop reasons, tracebacks, condition-order rotation, model settings, safer export behavior, single-run locking, additional tests, and the `docs/` research plan.
- The local working tree contains the reflection-enabled `v2-reflect` harness plus batch IDs, self-describing filenames, sequential local batch execution, and Single/Batch dashboard comparison.

Those capabilities should be combined and retested before the final submission. Session journaling and durable mid-game resume are designed but **not implemented**. Suspending a process with `SIGSTOP` is not durable session management.

---

## 1. Formal finite hypothesis representation compatible with LLM output

### Answer

The target hypothesis space is a Cartesian product of six finite rule dimensions:

\[
\mathcal{H}=H_P\times H_G\times H_M\times H_T\times H_I\times H_S.
\]

Each dimension has at most 16 alternatives and therefore occupies one four-bit register. Register order is `P, G, M, T, I, S`, giving a fixed 24-bit hypothesis index:

\[
x=16^5p+16^4g+16^3m+16^2t+16i+s.
\]

For example:

```text
(P0, G0, M0, T0, I1, S0)
= 0000 0000 0000 0000 0001 0000
```

The LLM should instantiate a finite vocabulary for each dimension from the observed frame. Each option must contain:

- a stable index;
- a short natural-language meaning;
- machine-checkable categorical predictions; and
- the actions and dimensions to which those predictions apply.

At scales where a dimension has fewer than 16 options, unused indices are invalid. Classical code uses a validity mask. A quantum implementation must specify whether invalid states are excluded during state preparation or marked by a reversible validity oracle, including all workspace and ancilla qubits.

### Current evidence

- `factorized.py` demonstrates six dimensions with two options each: 64 complete world models.
- The live ARC path still uses two to eight free-text hypotheses generated per turn.
- The local reflection-enabled harness preserves a structured belief ledger across turns, but this ledger is not the target fixed factorized space.

### Completion criterion

- Implement `encode`, `decode`, `valid_mask`, and natural-language rendering for variable `k_d ≤ 16`.
- Freeze the actual option vocabulary or the schema by which the LLM instantiates it.
- Demonstrate round-trip and invalid-state tests at every agreed scale.
- Resolve the inconsistent S1/S2 scale definitions and the duplicated H1/H2/H3 tuple examples in `TASK.md`.

**Status: partially implemented.**

---

## 2. Mathematically justified search and utility objective

### Answer

Let `O_t` be the evidence available through turn `t`. The posterior should be defined using an observation likelihood:

\[
P(h\mid O_t)\propto P(h)\prod_k P(o_k\mid h,a_k).
\]

A strict consistency indicator is a useful noiseless special case, but the integrated system should not assign zero probability merely because an LLM misinterprets one observation.

For deterministic predictions, the expected information gain of action `a` is the entropy of its predicted outcome distribution:

\[
p(o\mid a)=\sum_h P(h\mid O_t)\mathbf{1}[f(h,a)=o],
\]

\[
IG(a)=H[p(o\mid a)].
\]

The experimental objective should begin with pure information gain. Additional terms may be introduced only as separately switchable and justified ablations:

\[
U(a)=IG(a)+\lambda_g P(\text{progress}\mid a)-\lambda_c C(a)-\lambda_r R(a).
\]

This objective answers the central question: choose the action expected to distinguish the currently plausible world models, not merely the action preferred by the most confident hypothesis.

### Current evidence

- The controlled 64-world track implements exact information gain.
- The live ARC path uses a hand-tuned utility containing information gain, LLM-estimated progress and risk, novelty, outcome bonuses, and repetition penalties.
- In the saved live traces analyzed on remote `main`, exact classical selection chose the highest-utility action on 190/190 turns. The current encoded hypothesis register therefore does not materially affect classical action selection.

### Completion criterion

- Implement a compositional categorical predictor `f(h,a,O_t)` for the factorized space.
- Compare exact IG with hand-computed small examples.
- Define a noise-tolerant observation model.
- Run ablations showing the effect of every non-IG term.
- Demonstrate that changing the posterior over hypotheses can change the selected action.

**Status: correct in the controlled track; not yet implemented end to end for ARC.**

---

## 3. Working QAHS implementation in Python/Jupyter on a simulator

### Answer

The intended external interface is:

```python
select_action(
    space,
    predictions,
    observations,
    actions,
    backend,
) -> (action, score, diagnostics)
```

The integrated loop is:

```text
observe → instantiate/update vocabulary → compute posterior
→ score experiments → select action → execute → reflect → repeat
```

Termination must come from an environment-verified win, game over, explicit error, or exhausted action budget—not from an unsupported model declaration.

### Current evidence

- The Python ARC harness runs LLM-only, exact-classical QAHS, and simulated-QAOA QAHS.
- The reflection-enabled local work makes a second structured model call after each confirmed action and carries the resulting belief ledger into the next turn.
- The local dashboard can run one game or a sequential versioned batch and compare batch metrics by game and condition.
- The remote branch adds explicit stop reasons and retries idempotent model requests.
- The existing QAOA backend is a NumPy statevector simulation.

### Missing work

- The 16-option factorized ARC engine does not exist.
- QAE and Grover backends do not exist.
- The requested Jupyter notebooks do not exist.
- Remote Phase-0 robustness changes and local batch/versioning changes are not merged.
- Durable session journaling, pause, and resume do not exist.

### Completion criterion

- Reconcile the branches and pass the combined test suite.
- Implement the factorized search API and import it from notebooks rather than duplicating logic.
- Produce the five planned notebooks: representation, objective, quantum formulations, scaling, and ARC experiment.
- Add restart-safe experiment journaling before long batch studies.

**Status: working harness, incomplete target implementation.**

---

## 4. Justification for the selected quantum algorithm

### Answer

With only about four to seven available actions, selecting the largest score is trivial. The potentially expensive operation is evaluating outcome probability mass across a large posterior over hypotheses. This makes the primary target an **estimation** problem rather than an action-argmax problem.

- **Quantum amplitude estimation (QAE):** candidate primary method for estimating `p(o|a)`. Under coherent state-preparation and outcome-oracle access, QAE can obtain additive error `ε` with `O(1/ε)` oracle uses versus `O(1/ε²)` classical Monte Carlo samples.
- **Amplitude amplification/Grover search:** candidate method for finding or sampling hypotheses consistent with accumulated evidence.
- **QAOA/QUBO:** appropriate only when the decision itself is combinatorial, such as selecting a multi-step action plan or a maximum-a-posteriori world model.

These are query-complexity statements, not end-to-end speedup claims. State preparation, reversible prediction, oracle depth, repeated QAE calls across outcomes/actions, and error correction can remove any practical advantage.

### Current evidence

- QAOA is implemented over the current small hypothesis-action encoding.
- Its cost is separable by action except for a small belief tie-break, so the present circuit does not demonstrate a meaningful quantum search over interacting hypotheses and actions.
- No QAE or Grover oracle has been implemented.

### Completion criterion

- Specify reversible state-preparation, prediction, consistency, and validity oracles.
- Count their queries, gates, ancillas, and depth.
- Show QAE/Grover correctness against exact classical calculations at small scale.
- Retain QAOA only for an explicitly combinatorial formulation.

**Status: algorithm choice justified conceptually; primary quantum algorithm not implemented.**

---

## 5. Resource and scaling analysis

### Answer

The six hypothesis registers require up to 24 index qubits. Additional resources include:

- 2–3 action qubits;
- outcome and validity flags;
- QAE precision qubits;
- reversible predictor workspace; and
- any uncomputation ancillas.

Statevector memory alone is:

\[
2^n\times16\text{ bytes}
\]

for complex128 amplitudes. Thus 24 qubits require about 268 MB and 30 qubits about 17 GB. A complete QAE circuit with workspace may be too large for full-statevector simulation even when the 24-qubit hypothesis index alone fits.

The classical baseline must remain strong. A single float32 vector over `16^6` hypotheses is about 67 MB, although complete predictions and temporary arrays require more memory. Factorized marginalization may avoid full enumeration when an action depends on only a few dimensions.

### Current evidence

- `scaling.py` estimates statevector memory.
- No measured QAE/Grover circuit resources exist.
- The 35–45 logical-qubit estimate remains provisional.

### Completion criterion

For every scale and backend, report:

- hypothesis count;
- logical and transpiled qubits;
- ancillas;
- depth and two-qubit gates;
- shots, iterations, and oracle calls;
- objective error/regret;
- classical memory and compute; and
- noiseless versus noisy results.

**Status: preliminary estimates only.**

---

## 6. Correctness tests at small scale, then scaling toward 16.8 million

### Answer

Correctness must precede scaling:

- encode/decode round trips and invalid-index rejection;
- posterior normalization and update checks;
- `IG=0` when all hypotheses predict the same outcome;
- `IG=log₂(k)` for an even `k`-outcome split;
- quantum estimates within declared tolerance of exact values;
- Grover samples restricted to marked hypotheses; and
- QAOA approximation quality only on a justified combinatorial objective.

### Current evidence

- Remote `main` reports 26 tests after its Phase-0 additions.
- The divergent local working tree has 17 passing tests, including reflection parsing and versioned batch filenames.
- The controlled track validates encoding, QUBO/Ising energy agreement, QAOA smoke behavior, and the 64-world episode.

These counts cannot simply be added; the branches must first be reconciled and the combined suite rerun.

### Completion criterion

- One merged suite must pass in a clean clone.
- Exact classical results must be the oracle for small quantum instances.
- Scaling should proceed through agreed S1–S5 definitions, stopping where simulation is no longer feasible and reporting that boundary honestly.

**Status: good small-prototype coverage; target scaling tests missing.**

---

## 7. Experimental comparison with controlled inputs

### Answer

Required conditions:

1. LLM-only;
2. LLM + exact classical QAHS;
3. LLM + quantum QAHS;
4. random baseline; and
5. simple greedy baseline.

The conditions can share identical initial game, seed, action budget, model, prompts, vocabulary schema, and model settings. They **cannot share identical observations after their first different action**, because each policy creates its own trajectory. The correct experimental claim is matched initial conditions and an identical observation-processing pipeline.

Where two conditions independently reach an identical state and evidence history, shared caching can reuse the same proposal/reflection. Runtime must still separate model latency, cache hits, classical objective construction, and selector time.

### Current evidence

- A three-condition matched harness exists.
- Remote work rotates condition order across seeds to reduce order effects.
- Local work adds `batch_id`, `harness_version`, model/seed/budget metadata, self-describing filenames, sequential batch execution, and dashboard batch comparisons.
- Existing older runs and reflection-enabled runs are not directly comparable.
- The random baseline is not integrated, and the greedy baseline is not part of the live comparison.

### Completion criterion

- Merge robustness and batch/versioning features.
- Freeze prompts, model settings, factorized vocabulary, quantum settings, and exclusion rules.
- Use several seeds per game and record failures without silently rerunning only unfavorable conditions.
- Obtain Group 1's success definition before held-out evaluation.
- Report confidence intervals and paired comparisons across game/seed units.

**Status: useful experimental harness; target controlled study not ready.**

---

## 8. Experimental results comparing LLM-only with LLM + QAHS

### Answer based on completed evidence

The saved live ARC runs available before the new versioned batch work produced no complete game wins. The strongest recorded result was one completed level, reached in separate runs by LLM-only and exact classical QAHS. Some recorded failures were caused by network interruption and should be labeled as errors rather than unsuccessful reasoning attempts.

The current live results do **not** test the target factorized QAHS architecture. They test a free-text hypothesis generator plus a hand-tuned per-action utility and simulated QAOA selector. The controlled 64-world environment is useful for correctness but is too easy to distinguish the methods.

The newly started reflection-enabled batch is ongoing/paused and must not be included in completed results until its traces are finalized and validated.

### Defensible conclusion today

> The repository demonstrates an end-to-end experimental harness and small-scale quantum-search prototype. It currently provides no evidence for or against the target claim that quantum-assisted hypothesis search improves adaptive reasoning, and it provides no evidence of quantum computational advantage.

### Evidence still required

- Target factorized representation and objective integrated into ARC.
- Completed, version-consistent game/seed batches.
- Reproducible aggregation script—not hand-calculated totals.
- Paired statistics with confidence intervals.
- Search-quality and quantum-resource results separated from model/API latency.
- Ablations showing whether any gain comes from persistent reflection, information-gain selection, the quantum selector, or additional computation.

**Status: preliminary negative/inconclusive evidence; final experiment not run.**

---

## Overall completion summary

| Deliverable | Current assessment |
|---|---|
| Finite representation | Partial: 64-world prototype; target 16⁶ not built |
| Objective | Controlled IG exists; live target objective not integrated |
| Working implementation | End-to-end harness exists; target backends/notebooks missing |
| Quantum justification | Strong conceptual rationale; primary method missing |
| Scaling analysis | Preliminary memory estimates only |
| Correctness/scaling tests | Small-scale tests exist; merged/scaled suite missing |
| Controlled comparison | Three-condition harness and local batching exist on divergent branches |
| Final results | Inconclusive; target experiment not yet conducted |

The immediate engineering priority is to reconcile remote Phase-0 robustness with local batch/versioning work. The immediate research priority is to implement the factorized representation and exact classical information-gain baseline before adding a more ambitious quantum backend.
