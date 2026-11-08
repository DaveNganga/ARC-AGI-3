# QAHS V3.3 Collaborative Architecture

## Goal

QAHS V3.2 is a unilateral augmentation: the LLM proposes an action and QAHS can override it. V3.3 treats the LLM and QAHS as peer reasoning sources and adds an evidence-based arbitrator.

## Inputs

At each turn the LLM receives the same agent-visible observation, history, action space, and public 96-rule ontology as the baseline. QAHS receives the same agent-visible information. Neither system receives the hidden world tuple or the true symbol/class mapping.

## LLM hypothesis

The LLM may nominate one or more leading rules for P/G/M/T/I/S and a confidence value. If a rule is nominated in every family, V3.3 forms a complete 24-bit candidate world. Partial nominations are retained as partial evidence but are not treated as a complete world.

## QAHS hypothesis

QAHS maintains the factorized posterior over the 16 rules in each family, samples joint candidates over the implicit 16^6 space, and uses its posterior MAP world as its explicit candidate hypothesis.

## Evidence score

For a complete candidate world H, V3.3 computes:

`log_posterior(H) = log P(H | D_t)`

using the current factorized posterior. It also computes a bounded predictive-fit score from how well H predicts the observed action/transition history. The composite arbitration score is:

`0.78 * log_posterior + 0.22 * predictive_fit`

For the LLM candidate only, a bounded confidence term is added with weight `llm_confidence_weight` (default 0.20). The LLM's self-reported confidence never overrides empirical evidence by itself.

## Arbitration

V3.3 has three conceptual outcomes:

1. **Accept LLM**: the LLM's complete hypothesis is materially better supported than QAHS's candidate and its proposed action remains the selected action.
2. **Accept QAHS**: QAHS's candidate is better supported, or its action has the stronger evidence-based utility when the hypotheses are not close enough to justify a diagnostic experiment.
3. **Diagnostic experiment**: the two hypotheses are too close. V3.3 searches the four legal actions for an action whose predicted outcomes differ most between the LLM and QAHS hypotheses while retaining reasonable decision utility. That action is executed to generate new evidence.

This is deliberately asymmetric in authority: **confidence alone cannot make the LLM win, and QAHS cannot win simply because it is QAHS**.

## Frozen defaults

- arbitration margin: 0.15
- diagnostic threshold: 0.12
- LLM confidence weight: 0.20
- particles: 384
- evaluation particles: 64
- QAOA reps: 1
- QAOA shots: 256
- QAOA optimizer max iterations: 12
- planning horizon: 8
- beam width: 48
- max actions: 12

The V3.2 code and V3.2 production runner remain unchanged; V3.3 is a separate condition so the original experiment remains reproducible.
