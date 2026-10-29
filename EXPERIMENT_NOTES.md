# Experiment notes

## Status

These are engineering smoke tests on a controlled synthetic environment. They are not evidence for stronger AI or quantum advantage.

## 64-world classical run

- Conditions: deterministic policy proxy and exact classical QAHS.
- Hidden worlds: all 64 combinations of the six binary rule dimensions.
- Result: both conditions solved 64/64 worlds in seven actions on average.
- Interpretation: the factorized representation, observation updates, exact objective, and termination logic are internally consistent. The environment is deliberately easy and therefore cannot distinguish the conditions.

## QAOA smoke runs

- QAOA configuration: statevector simulation, depth `p=2`, four COBYLA restarts, 100 optimizer evaluations per restart, and 1,024 shots per turn.
- Development worlds 0-3: 4/4 solved, 6.75 actions on average.
- Separate smoke worlds 4-7: 4/4 solved, seven actions on average.
- The non-integer average on worlds 0-3 came from one correct early guess. Four worlds are far too few to treat that as a performance improvement.
- An earlier low-budget QAOA configuration was unreliable. The formulation was corrected to encode observed rule values as explicit consistency penalties instead of weak tie-break terms. The current results must therefore be treated as post-development smoke tests, not held-out evidence.

## Remaining experimental requirements

1. Integrate a real LLM and freeze its prompt and output schema.
2. Replace the controlled rule-probe environment with several genuine ARC-like tasks.
3. Pre-register QAOA depth, optimizer budget, shots, penalty weights, seeds, and primary metrics.
4. Compare an actual LLM-only action choice against exact classical QAHS and QAOA-assisted QAHS using identical inputs.
5. Run enough held-out tasks and seeds to report uncertainty.
6. Measure circuit depth and transpiled resources in Qiskit; the current statevector engine estimates logical resources only.

