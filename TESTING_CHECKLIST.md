# QAHS Prototype Testing Checklist

Use this checklist in order. The first two sections test the software you already have. The later sections turn it into a defensible research experiment.

## 1. Confirm the current code works

- [ ] From the `QAHS-Prototype` folder, run `PYTHONPATH=src python3 -m unittest discover -s tests -v`.
- [ ] Confirm all tests pass.
- [ ] Run `PYTHONPATH=src python3 -m qahs.cli --backend classical --hidden-rule red_is_switch --no-save`.
- [ ] Run `PYTHONPATH=src python3 -m qahs.cli --backend qaoa --hidden-rule red_is_switch --no-save`.
- [ ] Confirm both runs choose red first, observe that the door opens, then choose green.
- [ ] Repeat the small demo with each of the four hidden rules and record success, turns, and chosen actions.
- [ ] Reproduce the current limitation: all three backends succeed on the switch and goal worlds but fail on the hazard and does-nothing worlds. Treat this as a scoring/stopping-rule issue to fix, not a QAOA-specific failure.

## 2. Inspect the loop in the dashboard

- [ ] Create a local demonstration with `.venv/bin/qahs-arc --source demo --condition compare --max-steps 6`.
- [ ] Start the frontend with `bash run_dashboard.sh` and open `http://127.0.0.1:3000`.
- [ ] Check that exact classical and QAOA receive the same action scores and objective.
- [ ] Confirm the grouped chart and details show all three conditions.
- [ ] Confirm the private trace is in `output/arc_runs/` and its sanitized copy is in `frontend/public/runs/`.

## 3. Reproduce the controlled 64-world benchmark

- [ ] Run `PYTHONPATH=src python3 -m qahs.benchmark_cli --conditions policy_proxy classical --worlds 64 --output output/experiments/reproduction_classical_64.json`.
- [ ] Confirm both conditions solve 64 of 64 worlds.
- [ ] Confirm the expected result is about seven actions: six probes and one solution attempt.
- [ ] Run a small QAOA check with `PYTHONPATH=src python3 -m qahs.benchmark_cli --conditions qaoa --worlds 8 --output output/experiments/reproduction_qaoa_8.json`.
- [ ] Record solve rate, actions, wall time, shots, optimizer evaluations, and seed.
- [ ] Do not describe a difference from eight worlds as an advantage.

## 4. Freeze a fair experiment before collecting final results

- [ ] Write down the primary claim before testing: for example, QAOA-assisted search reduces actions to solution under the same hypothesis and action budget.
- [ ] Use the same environment states, candidate hypotheses, candidate actions, scoring rule, and stopping rule in every condition.
- [ ] Compare at least: LLM-only choice, greedy classical, exact classical, and QAOA-assisted search.
- [ ] Freeze QAOA depth, restarts, optimizer limit, shots, penalty weights, and random seeds.
- [ ] Separate development tasks from held-out test tasks.
- [ ] Select primary metrics: solve rate and actions to solution.
- [ ] Select secondary metrics: wall time, number of model calls, quantum shots, optimizer evaluations, and logical circuit resources.
- [ ] Repeat stochastic conditions across multiple seeds and report uncertainty, not only averages.

## 5. Add a real language model

- [ ] Install the optional dependency with `python3 -m pip install -e '.[llm]'`.
- [ ] Set `OPENAI_API_KEY` and `OPENAI_MODEL` in the shell. Never store the key in the repository.
- [x] Wire `OpenAIJSONCaller` into the experiment runner. It is used by `qahs-arc --source arc` and the local dashboard runner; the 64-world experiment does not use an LLM.
- [ ] Freeze the prompt and JSON schema before held-out testing.
- [ ] Log the model's proposed hypotheses and actions for every turn.
- [ ] Reject or repair malformed output consistently across all conditions.
- [ ] Add a true LLM-only baseline that chooses an action without QAHS.

## 6. Replace the easy teaching world with a meaningful task

- [ ] Implement an ARC or other interactive environment through the adapter contract.
- [ ] Ensure the hidden answer is available to the evaluation harness but never shown to the agent.
- [ ] Remove direct probes that simply reveal one rule bit at a time.
- [ ] Require actions whose outcomes can support several competing hypotheses.
- [ ] Make the scoring rule depend only on information available to the agent.
- [ ] Include tasks where a greedy next action is sometimes worse than a longer-term plan.
- [ ] Include enough held-out tasks to distinguish the methods.

## 7. Test the quantum formulation itself

- [ ] Verify every valid bitstring decodes to one hypothesis and one action.
- [ ] Verify invalid action encodings receive a penalty larger than any possible reward.
- [ ] Verify the QUBO and Ising energies match on random bitstrings.
- [ ] Inspect whether the objective contains meaningful hypothesis-action interaction terms. The current 64-world QUBO mostly selects a hypothesis and an action independently.
- [ ] Compare QAOA's sampled candidates with the exact minimum on small instances.
- [ ] Track the probability assigned to optimal and near-optimal candidates.
- [ ] Increase problem size gradually and record when simulation becomes impractical.

## 8. Add Qiskit evidence for the course deliverable

- [ ] Rebuild the QUBO or Ising objective as a Qiskit cost operator.
- [ ] Run Qiskit QAOA on a noiseless simulator and compare with the current NumPy simulator.
- [ ] Transpile the circuit and report logical qubits, physical qubits if applicable, depth, one-qubit gates, and two-qubit gates.
- [ ] Run a noise model and compare solve quality with the noiseless result.
- [ ] If hardware access is available, use only a small frozen instance and report queue, calibration, shots, and errors.

## 9. Decide what the results support

- [ ] Claim only software correctness if the loop works on the teaching environments.
- [ ] Claim better search only if held-out results beat matched classical baselines on the primary metrics.
- [ ] Report cases where QAOA is slower or less reliable.
- [ ] Separate quantum-assisted reasoning evidence from quantum speedup. They are different claims.
- [ ] Preserve code, prompts, seeds, raw traces, and analysis scripts needed to reproduce every reported figure.

## 10. Run the ARC-AGI-3 comparison harness

- [ ] Run `bash setup_arc.sh` if `.venv` does not exist.
- [ ] Run `bash run_dashboard.sh` and open `http://127.0.0.1:3000`.
- [ ] Complete the local demonstration before adding credentials.
- [ ] Confirm the control trace uses `llm_recommendation` and reports `quantum_used: false`.
- [ ] Confirm the QAHS trace uses `qaoa_statevector` and reports `quantum_used: true`.
- [ ] Export `OPENAI_API_KEY`, `OPENAI_MODEL`, and `ARC_API_KEY` in the same terminal used to launch the experiment.
- [ ] Start the first live comparison with `.venv/bin/qahs-arc --source arc --game ls20 --condition compare --seed 0 --max-steps 4`.
- [ ] Confirm both sessions use the same game ID, seed, action budget, and model configuration.
- [ ] Inspect the first-turn proposal and verify that identical state/history inputs were reused across conditions.
- [ ] Save the trace and record any API, parsing, invalid-action, or environment errors.

