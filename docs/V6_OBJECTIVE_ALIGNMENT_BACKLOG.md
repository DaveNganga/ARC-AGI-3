# Harness v6: Objective Alignment Backlog

**Planned harness name:** `v6-objective-alignment`  
**Status:** Implemented as `v6-objective-alignment`; pending a short smoke run and matched experimental validation against v5.

## Motivation

The v5 selector can favor hypotheses that reliably explain mechanics without helping the agent discover the objective. In `ar25`, repeated movement explanations became highly confident while a low-confidence but goal-relevant hypothesis received little attention. V6 will preserve useful exploration while making actual progress, decisive goal tests, and escape from stagnation more important.

## Planned work

### 1. Separate mechanism knowledge from goal relevance

- Score each hypothesis on both explanatory reliability and relevance to progress.
- Treat established mechanics as retained background knowledge rather than endlessly valuable experiments.
- Add explicit objective terms for goal relevance and expected progress.
- Preserve the distinction in trace diagnostics.

### 2. Reward goal–action alignment

- Measure whether an action directly tests its hypothesis's `goal_condition`.
- Reward actions predicted as `PROGRESS` or `TERMINAL` when they are concrete and falsifiable.
- Reward contact, interaction, boundary-crossing, or target-directed tests when supported by the grid and hypothesis.
- Do not grant a goal bonus to an unrelated action merely because its hypothesis mentions a goal.

### 3. Saturate confidence

- Prevent confidence near `1.0` from dominating the objective indefinitely.
- Cap or nonlinearly transform confidence for selection purposes while retaining the original value in the trace.
- Increase the cost of repeatedly confirming an already-established mechanism.

### 4. Add stagnation-driven goal weighting

- Track turns since the last increase in `levels_completed` or terminal success.
- Begin raising goal weight after approximately 15% of the action budget has been spent pursuing one goal without measurable impact.
- Track stagnation per canonical goal rather than treating unrelated exploration as one undifferentiated counter.
- Gradually raise the weight of alternative goal-directed and decisive tests during prolonged non-progress.
- Reset or reduce the stagnation pressure after genuine progress.
- Keep the schedule deterministic and record its value per turn.

### 5. Classify hypothesis roles

- Assign hypotheses to a compact role such as `mechanism`, `goal`, `obstacle`, `interaction`, or `terminal`.
- Have the model propose the role and have Python validate or normalize it from the structured hypothesis fields.
- Penalize excessive consecutive selections from the same role when no progress occurs.
- Avoid a hard quota that forces a weak goal hypothesis when strong evidence indicates a necessary mechanism test.

### 6. Reward decisive falsification

- Value an uncertain hypothesis when a low-cost action could strongly confirm or reject it.
- Combine possible progress, information gain, falsifiability, and action cost.
- Prefer one decisive goal test over several minor confirmations of known behavior.

### 7. Propagate progress credit backward

- Group actions into a **goal episode**: the actions performed while the controller is actively pursuing the same canonical goal.
- When `levels_completed` increases, credit the successful goal and every action associated with that goal episode.
- Weight credit more strongly for actions nearer the success so a long episode does not imply that every early action was equally necessary.
- Exclude unrelated exploration and actions assigned to a different goal family.
- Carry reusable strategy evidence into the next level without assuming that the new layout is identical.
- Record the credit assignment in the saved diagnostics.

### 8. Semantically deduplicate hypotheses

- Group hypotheses that express the same rule despite changed wording or IDs.
- Use structured fields first; use normalized text only as supporting evidence.
- Ensure one semantic mechanism family cannot occupy a disproportionate share of the search space.
- Preserve original hypotheses in the JSON while constructing the selector from canonical families.

## Level-boundary behavior

V6 should treat a level-counter increase as a reasoning boundary:

1. Save the successful short policy and its evidence.
2. Preserve general mechanics that remain supported.
3. Mark layout-specific beliefs for revalidation.
4. Give the successful policy one transfer attempt on the new level.
5. Replan normally if transfer fails.

## Validation order

1. Finish and analyze both v5 seed batches.
2. Implement the revised scoring terms in the classical selector first.
3. Run deterministic objective-ranking tests and targeted regressions.
4. Compare classical v5 and v6 behavior on matched fixtures.
5. Apply the identical validated cost model to QAOA.
6. Change circuit depth, initialization, or mixer only if the revised objective is useful but quantum approximation remains inadequate.

## Efficiency track

V6 should reduce repetitive model work without using aggressive early termination. Implement and evaluate these changes independently where practical so their effect on behavior remains measurable.

### 1. Adaptive 12–16-plan catalog

- Replace the fixed 32-plan set with a balanced catalog of goal-directed, information-seeking, recovery, and context-specific interaction plans.
- Preserve diversity across physical actions and canonical hypothesis roles.
- Record which pruning rule admitted or rejected each candidate.

### 2. Semantic hypothesis deduplication

- Reuse the canonical hypothesis families planned for objective alignment.
- Remove duplicate prediction work while retaining original model statements in the trace.
- Ensure deduplication happens before catalog scoring and quantum encoding.

### 3. Initial full grid followed by deltas

- Send the image, summary, and complete lossless grid at the start of a condition.
- Send changed coordinates and affected regions on routine turns.
- Refresh the complete grid after a level transition, major visual change, uncertainty about accumulated state, or a fixed verification interval.
- Keep the complete local observation in every saved trace.

### 4. Conditional second-action execution

- Execute the second planned action without another model call only when the first result matches its prediction, no level or terminal transition occurred, the action remains valid, and risk is below a configured threshold.
- Replan immediately when any validation fails.
- Record whether the second action was accepted, rejected, or invalidated and why.

### 5. Python routine updates

- Update repetition, confidence evidence, stagnation, changed cells, and progress bookkeeping locally for predictable routine outcomes.
- Do not let deterministic updates invent new hypotheses or silently reinterpret unexpected evidence.
- Preserve periodic model review.

### 6. Event-triggered Luna calls

- Request new model reasoning on a level transition, terminal state, novel or unexpected frame, goal contact, obstacle discovery, failed prediction, completed plan, or stagnation threshold.
- Allow routine validated policy steps between events.
- Record the event that caused every model call.

### 7. Strategy reset instead of early termination

- Preserve the complete action budget unless the environment becomes terminal.
- When stagnation crosses its threshold, retire or down-weight the current goal episode, increase alternative-goal pressure, and request a fresh strategy.
- Retain general mechanics while forcing a materially different goal test.

### Efficiency evaluation

Compare v6 with v5 on model calls, input tokens, wall time, actions, progress, selector behavior, and missed-event rate. Runtime comparisons involving parallel conditions must be reported separately because concurrent resource contention makes them unsuitable as evidence of quantum speedup.


## Required diagnostics

Every v6 turn should record:

- hypothesis role and canonical family;
- mechanism score and goal-relevance score;
- goal–action alignment;
- effective confidence after saturation;
- stagnation count and dynamic goal weight;
- falsification value;
- progress credit received;
- deduplication mapping;
- final term-by-term cost contribution.

## Evaluation

V6 should be compared against v5 using matched games, seeds, model, action budgets, and condition order. Primary evidence remains solved games and levels completed. Secondary evidence includes actions to progress, model calls, selector regret, repeated semantic families, overrides, and goal-directed test frequency.

V6 is successful if it improves at least one agreed practical outcome without hiding a material regression in the others. Exact optimization of the revised objective is not by itself evidence of better ARC performance.

## Decisions recorded

- Goal stagnation begins at approximately 15% of the action budget spent on one canonical goal without measurable impact.
- Hypothesis roles are model-supplied and Python-validated.
- Progress credit follows the complete successful goal episode, with recency weighting and unrelated actions excluded.
- A successful policy receives one transfer attempt on the next level before normal replanning resumes.

## Remaining implementation decisions

- Exact confidence transformation: hard cap, logarithmic saturation, or diminishing-return penalty.
- Maximum goal-weight increase after stagnation begins.
- Decay function and optional maximum length for goal-episode credit.
- Similarity threshold and structured fields used for semantic deduplication.
