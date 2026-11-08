# Paper sections: Methodology B–F, Results and Discussion, abstract edits, references

Draft text for the team paper, matching the results in this folder and in `frontend/public/runs/`.
Citation numbers continue the paper's IEEE list ([1]–[6] are the team's existing references).

## Methodology

**B) Hidden-Rule Test Environment**

To evaluate the framework under controlled conditions, we used an ARC-AGI-like hidden-rule benchmark [1], [11].
In this benchmark, an agent interacts with a small grid world whose rules are hidden. Each world is defined by one
rule from each of six families: player control, goal, movement, topology, interaction, and sequence. Each family
has 16 possible rules, giving 16⁶ = 16,777,216 = 2²⁴ possible worlds. The agent only sees the grid, the objects on
it, and the available actions (UP, DOWN, LEFT, RIGHT). It is never told the active rules or what each object
represents. The true rules are kept only for scoring after each episode.

To ensure a fair comparison, we fixed all benchmark worlds, random seeds, and action budgets in a manifest before
running the experiments. This allowed us to test every model and condition on the same episodes. We also verified
that every clean world could be solved within the 12-action limit.

Each condition was tested on 18 episodes, covering four aspects of performance:

- **Generalization:** Five previously unseen worlds.
- **Adaptation:** Three episodes in which the hidden rules change at turn 4.
- **Transfer:** One source episode followed by three matched pairs of episodes. Each target world is attempted
  twice: first without prior experience (cold), and then with knowledge gained from the source episode (warm).
- **Robustness:** Three episodes in which 20% of observations are perturbed.

Together these give 5 + 3 + 1 + 6 + 3 = 18 episodes. We calculated autonomy using the five generalization episodes
rather than running a separate set of tests.

**C) Quantum-Assisted Hypothesis Search (QAHS)**

QAHS divides the work between a language model and a search module. At each turn the LLM receives the current
observation and recent evidence. It proposes an action together with candidate rules or hypotheses. QAHS then
maintains a probability distribution over the hypothesis space and selects the action used. The QAHS implementation was developed with the AI-assisted
development and assurance workflow of Hartwell et al. [12].

In the hidden-rule benchmark, QAHS maintains a factorized Bayesian posterior over the six rule families, with 384
joint particles for interactions between families. Candidate worlds are sampled by the Quantum Approximate
Optimization Algorithm (QAOA) [7] on a 24-qubit register, four qubits per family. For each family, a quadratic
approximation of the negative log-posterior is compiled into an Ising cost Hamiltonian. The circuit uses one QAOA
layer (p = 1) and 256 shots, with parameters tuned by COBYLA [10] for up to 12 iterations. It is executed on the
Qiskit Aer matrix-product-state simulator [8]. Sampled worlds are then ranked by the exact posterior. The agent
moves through explore, goal-discrimination, and exploit modes. It plans up to 8 actions ahead with a beam width of
48, re-planning after each action, and the LLM's proposal receives a weight of 0.35.

In the ARC-AGI-3 experiments, we used the v12 structured-world-models harness. At each turn the LLM proposes 6–10
competing complete hypotheses, each predicting the outcome of every available action. The harness classifies the
observed outcome and updates hypothesis probabilities. QAHS then selects the ordered three-action plan that best
separates the remaining hypotheses, executes its first action, and re-plans. Two selectors optimize the identical
objective over identical candidates: classical QAHS uses an exact search that always returns the lowest-cost plan,
and quantum QAHS uses QAOA with depth p = 2, 3 restarts, 80 COBYLA iterations, and 512 shots, run on an exact
state-vector simulator and seeded deterministically per turn.

**D) Experimental Conditions**

Each experiment compared three conditions on the same episodes: (1) LLM-only, in which the model's recommended
action is executed directly; (2) classical QAHS, in which the model's hypotheses are passed to the exact classical
selector; and (3) quantum QAHS, in which the same hypotheses are passed to the QAOA selector. The LLM-only
condition isolates native model reasoning. Comparing classical with quantum QAHS separates the effect of hypothesis
search from the effect of the quantum algorithm itself. In ARC-AGI-3, the order of the three conditions rotated
with the seed. All conditions shared the same action budget, stopping rules, and candidate actions.

**E) Mapping the Framework onto Measurements**

The criteria in Table 1 were operationalized as five scores between 0 and 1, combined into a Strong AI Index (SAI)
as their equal-weight mean. In the hidden-rule benchmark:

- **Generalization (G):** the solve rate on held-out worlds.
- **Adaptation (A):** the mean of max(0, 1 − L/6), where L is the number of turns from the hidden change to the
  first new grid-changing action (L = 6 if none occurs).
- **Autonomy (Au):** the mean of (NER + (1 − RAR) + (1 − NCAP)) / 3, where NER is the novel-experiment rate, RAR
  the redundant-action rate, and NCAP the persistence of repeated no-change actions.
- **Transfer (T):** each cold/warm pair scores 1 if only the warm attempt solves, 0 if only the cold attempt
  solves, 0.5 if both fail, and 0.5 + 0.5 · (cold actions − warm actions) / cold actions if both solve.
- **Robustness (R):** min(1, perturbed solve rate / clean solve rate), and 0 when no clean world is solved.

For ARC-AGI-3, scores were derived from each run's trace. G is levels completed divided by the levels required to
win; A is measured after each level transition as max(0, 1 − d/N), where d is the number of turns until the next
externally useful action and N is the episode length. An action is externally useful if it completes a level (full
credit) or produces a meaningful refutation (half credit): the hypotheses disagreed about the selected action's
outcome, the observed outcome matched one of their predictions, and the belief state was revised. Au is the
adaptive-recovery rate: the share of meaningful refutations later followed by a different, externally useful action. T and R are set to 0 because a
single game run contains no matched transfer or perturbation pair. Exploration and hypothesis testing (Table 1) is
reflected in the autonomy components and the recorded hypothesis catalogs but is not scored separately. All scores
were recomputed from per-episode data and checked against the stored reports, and differences between conditions
were evaluated with paired exact two-sided sign tests.

**F) Models and Configuration**

Four models were evaluated: Qwen3:8B [9], run locally through Ollama with temperature 0 and its output fields
enforced by a JSON schema; GPT-6 Luna, through the OpenAI Responses API; and Claude Opus 5.5 and Claude Haiku 5.5,
through the Anthropic Messages API with structured outputs. All API models ran at medium reasoning effort, and every
model received the same prompts and the same reply schema. For Claude, the output limit was 64,000 tokens, because
its reasoning counts toward that limit; an earlier 16,000-token limit truncated responses and was excluded from
comparisons.

Three experiments were run: (1) the hidden-rule benchmark with Qwen3:8B and Claude Opus 5.5, all 18 episodes in
each condition; (2) the ARC-AGI-3 game LP85 with GPT-6 Luna, seeds 6–31, a 40-action budget, and the v12 structured-world-models harness;
and (3) the same LP85 configuration with Claude Haiku 5.5 on seeds 6–31, plus Claude Opus 5.5 on seed 6.

## Results and Discussion

**A) Hidden-Rule Benchmark**

Table 2 reports the five capability scores and the Strong AI Index (SAI) for Qwen3:8B [9] and Claude Opus 5.5, each
with and without QAHS, on the 18-episode benchmark. QAHS raised the SAI of the smaller model from 0.400 to 0.475, but
lowered the SAI of Claude Opus 5.5 from 0.718 to 0.459. Most of each change comes from a few episodes. For Qwen, QAHS
added one generalization solve (G: 0/5 to 1/5) and changed one transfer pair from a cold-only solve to a double
failure (T: 0.333 to 0.500). For Claude, 0.200 of the 0.259 decrease comes from robustness: Claude alone solved 1 of
3 clean worlds and 2 of 3 perturbed versions (R = 1.000), whereas with QAHS it solved no clean world, so R is 0 by
definition. No difference between conditions was statistically significant (paired exact sign test, p ≥ 0.625 for
Qwen and p = 1.0 for Claude in every capability).

| Condition | G | A | Au | T | R | SAI |
|---|---|---|---|---|---|---|
| Qwen3:8B | 0.000 (0/5) | 1.000 | 0.667 | 0.333 | 0.000 | 0.400 |
| Qwen3:8B + QAHS | 0.200 (1/5) | 0.944 | 0.729 | 0.500 | 0.000 | 0.475 |
| Claude Opus 5.5 | 0.200 (1/5) | 0.944 | 0.944 | 0.500 | 1.000 | 0.718 |
| Claude Opus 5.5 + QAHS | 0.000 (0/5) | 1.000 | 0.754 | 0.542 | 0.000 | 0.459 |

*Table 2: Capability scores on the hidden-rule benchmark (18 episodes per condition).*

**B) ARC-AGI-3 Game LP85**

Table 3 summarizes the LP85 experiments [1] on the v12 structured-world-models harness, with 26 matched seeds (6–31)
and a 40-action budget. Claude Haiku 5.5 completed the first level in 34 of 78 episodes, against 9 of 78 for GPT-6
Luna. Haiku progressed on 12, 12, and 10 seeds in the LLM-only, classical, and quantum conditions; GPT-6 Luna
progressed on 2, 3, and 4 seeds. No run progressed beyond the first of eight levels. Claude Opus 5.5, run on seed 6
only, completed no level in any condition and is reported separately because it is not part of the matched sample.

| Model | Condition | Seeds with a level | Levels per 100 actions | SAI proxy |
|---|---|---|---|---|
| GPT-6 Luna | LLM-only | 2 / 26 | 0.336 | 0.194 |
| GPT-6 Luna | Classical QAHS | 3 / 26 | 0.326 | 0.216 |
| GPT-6 Luna | Quantum QAHS | 4 / 26 | 0.436 | 0.225 |
| Claude Haiku 5.5 | LLM-only | 12 / 26 | 1.515 | 0.288 |
| Claude Haiku 5.5 | Classical QAHS | 12 / 26 | 1.456 | 0.289 |
| Claude Haiku 5.5 | Quantum QAHS | 10 / 26 | 1.172 | 0.273 |
| Claude Opus 5.5 | LLM-only | 0 / 1 | 0.000 | 0.183 |
| Claude Opus 5.5 | Classical QAHS | 0 / 1 | 0.000 | — |
| Claude Opus 5.5 | Quantum QAHS | 0 / 1 | 0.000 | 0.192 |

*Table 3: LP85 results on 26 matched seeds (v12 structured-world-models harness, seeds 6–31, 40 actions; Claude Opus
5.5 is shown separately on seed 6 only). The SAI proxy is computed from the trace of each run with the dashboard's definitions; it is
reported for the LLM-only and quantum conditions, and fixes T and R at 0.*

Of Haiku's eleven level completions under QAHS on these seeds, seven directly followed an action in which the selector
chose an alternative to the model's recommended action: four under classical QAHS and three under quantum QAHS. For
GPT-6 Luna, one of its three QAHS completions followed such an override, under quantum QAHS. Because classical
selection produced the same pattern, these transitions show the value of hypothesis-driven selection in general, not
of QAOA specifically.

Paired seed comparisons did not establish a selector advantage. For Luna, quantum QAHS outperformed LLM-only on
three seeds and underperformed it on one (paired exact sign test, p = 0.625); classical QAHS had three wins and two
losses (p = 1.0). For Haiku, quantum QAHS had four wins and six losses (p = 0.754), while classical QAHS had four wins
and four losses (p = 1.0). Of Haiku's 22 QAHS level completions, 15 directly followed an action the selector chose
over the model's recommendation (7 of 12 under classical QAHS and 8 of 10 under quantum QAHS). These are behavioral
attributions for the executed action, not counterfactual proof that the model's rejected action would have failed.

**C) Discussion**

Across both environments, the model had a larger effect than the selection method. On the 26 matched LP85 seeds,
Haiku produced progress in 34 of 78 episodes, compared with 9 of 78 for GPT-6 Luna. Within Haiku, classical QAHS tied
LLM-only at 12 of 26 progressing seeds and quantum QAHS trailed at 10 of 26. Within Luna, quantum QAHS led at 4 of 26,
followed by classical QAHS at 3 and LLM-only at 2, but the paired differences were not statistically significant.
On the hidden-rule benchmark, QAHS helped the smaller
model and hindered the larger one, but each change rests on one to three episodes. Quantum and classical QAHS
performed comparably, as expected. Both optimize the same objective over the same candidates, and the classical
selector is exact, so QAOA [7] can match it but cannot exceed it except through sampling randomness. The present
results therefore do not demonstrate a quantum advantage. They indicate that structured hypothesis selection can
change agent behavior: it produced longer episodes and progress after overriding the model's preferred action.
As in [12], passing runs and recomputed scores show that the code executes and that the reported values are
reproducible, not that the implementation is behaviorally correct in every case. An early advantage for QAHS on Haiku
(seeds 11–15) disappeared as the sample grew, illustrating how small samples can suggest effects that do not hold.
Testing whether QAHS improves performance on other games, and separating the search architecture from QAOA with a
classical control using randomized tie-breaking, remain future work.

## Abstract edits

Replace "Our results provide evidence that QAOA-assisted selection can strengthen AI behavior." with:

> Results were mixed: QAHS improved the composite score for the smaller Qwen model but not for Claude, quantum and
> classical selection performed comparably, and on LP85 the choice of language model mattered more than the
> selection method.

Replace "In particular, quantum QAHS sometimes produced greater exploratory diversity, more progress per executed
action, and successful transitions following alternatives to the language model's preferred action." with:

> In particular, QAHS sometimes produced progress following alternatives to the language model's preferred action,
> although classical and quantum selection performed comparably.

## References to add after [6]

[7] E. Farhi, J. Goldstone, and S. Gutmann, "A quantum approximate optimization algorithm," arXiv:1411.4028, Nov. 2014.

[8] A. Javadi-Abhari et al., "Quantum computing with Qiskit," arXiv:2405.08810, May 2024.

[9] A. Yang et al., "Qwen3 technical report," arXiv:2505.09388, May 2025.

[10] M. J. D. Powell, "A direct search optimization method that models the objective and constraint functions by
linear interpolation," in *Advances in Optimization and Numerical Analysis*, S. Gomez and J.-P. Hennart, Eds.
Dordrecht, The Netherlands: Kluwer, 1994, pp. 51–67.

[11] F. Chollet, "On the measure of intelligence," arXiv:1911.01547, Nov. 2019.

[12] J. Hartwell et al., "AI developing AI: Can AI effectively and securely develop AI-based tools that really work?"
in *Proc. IEEE Military Communications Conference (MILCOM)*, 2026, doi: [DOI].
