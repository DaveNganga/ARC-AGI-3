# Harness v7: Grounded Goals Backlog

**Planned harness name:** `v7-grounded-goals`  
**Status:** Core implementation complete as `v7-grounded-goals`; pending live smoke validation and threshold calibration.

## Implemented core

- Structured grounded goals and cross-turn canonical goal identities.
- Goal-aware click catalog seeding from prior grounded targets.
- Animation-frame preservation and meaningful/cosmetic change classification.
- Candidate coverage, compatibility, exact-optimum, and complete candidate-table diagnostics.
- Evidence-margin override safety, grounded-policy commitment, cooldown pressure, and optional shadow mode.
- Override categories, intervention outcomes, and per-episode intervention summaries.
- Recorded model, effort, harness flags, input representation, and resumable experiment configuration.
- Deterministic regressions for semantic goal continuity and weak-override blocking.

## Purpose

V6 improved objective weighting and reduced repetitive reasoning, but the next harness must make goals stable, executable, and visibly connected to game progress. V7 should distinguish a promising objective from a concrete sequence the available ARC actions can actually perform.

## Goal representation and continuity

1. Replace prose-only goals with structured grounded goals: target component, color/shape, current bounds, destination or desired state, proposed action, prerequisites, and partial-success signal.
2. Give semantically equivalent goals a stable cross-turn identity. Match new proposals to a canonical goal ledger instead of hashing the current wording.
3. Track ordered multi-action goal episodes and assign progress credit to the actions associated with the successful goal, with stronger credit near the success.
4. Preserve productive interaction policies across turns and level boundaries while requiring revalidation when the layout changes.

## Candidate generation and execution

5. Diagnose catalog coverage on every turn: record whether the hypothesized target, action type, and destination are represented by executable candidates.
6. Generate click candidates around goal-relevant components, boundaries, and predicted interaction points rather than relying primarily on generic exploration.
7. Persist productive clicks or movements when the observed change matches the prediction; stop when progress, contradiction, or a meaningful event requires replanning.
8. Handle predictable interaction responses locally and request Luna/Sol reasoning only after meaningful events, uncertainty, or failed predictions.
9. Distinguish meaningful state changes from cosmetic animation so visual activity does not receive false progress credit.
10. Preserve the full relevant animation/frame stack for reasoning instead of exposing only the final frame when intermediate states carry causal evidence.

## Selector and search diagnostics

11. Separate overrides into goal-driven, evidence-driven, diversity/recovery, and approximation-driven categories.
12. Record both the exact classical objective optimum and the selected quantum solution, including term-level differences and whether either choice was executable.
13. Add a compatibility screen before selection: down-weight goals whose required interaction is unavailable through the current ARC action interface.
14. Add explicit diagnostics for goal identity, catalog coverage, target grounding, action feasibility, meaningful-change classification, and goal-episode credit.

## Override safety and model deference

The first GPT-6 Sol medium run on `ar25`, seed 0, provides the motivating case. The plain LLM completed one level at action 25 after identifying an exact gray-to-yellow alignment plan. Classical and quantum QAHS completed no levels while overriding the model recommendation on most turns. This does not establish general superiority for the LLM, but it shows that a selector can destroy a useful grounded policy even when it optimizes its encoded objective successfully.

15. Require an evidence margin before QAHS overrides a grounded model recommendation. An objective-score difference alone is insufficient.
16. Defer to a model plan when its target, desired state, and executable action are concrete and recent actions are changing the state as predicted.
17. Add a short policy-commitment window for validated multi-step movement. Reconsider it after contradiction, no change, level transition, terminal state, or target contact.
18. Add override cooldown and budget controls so repeated interventions cannot continually fragment one coherent plan without new evidence.
19. Run QAHS in optional shadow mode: record what classical and quantum would select while executing the model plan. Use this to measure counterfactual intervention quality before granting control.
20. Record intervention regret: whether an override improved progress, reduced estimated goal distance, produced useful information, or merely delayed the model plan.
21. Compare the selected action with the model recommendation using grounded outcome proxies such as target distance and predicted state change, while keeping actual level progress authoritative.
22. Treat post-transition replanning separately from ordinary overrides. A newly completed level should immediately end the previous commitment window and permit one grounded transfer attempt.

## Experiment controls

23. Make model and reasoning effort selectable for local runs and record both values in every episode, comparison, manifest entry, and resumable job.
24. Keep model, effort, game, seed, action budget, harness version, and condition order fixed within each matched comparison.
25. Improve paused-session discovery so an interrupted batch remains visible and resumable after the dashboard or runner restarts.

## Evaluation support

26. Save the complete candidate table, model recommendation, selector choice, objective terms, and override justification so a completed trace can be replayed without another model or ARC call.
27. Put major V7 behaviors behind recorded experiment flags, allowing grounded goals, commitment windows, override safety, and shadow mode to be ablated independently.
28. Report intervention precision: the share of overrides that improve progress or a validated goal-distance proxy. Report harmful and neutral overrides separately.
29. Compare against a model-deference condition in addition to plain LLM, classical QAHS, and quantum QAHS when evaluating the override policy.

## Deferred

- Grover-style search until grounded candidate generation and the objective function are validated.
- Parallel condition execution because it raises local resource and crash risk.
- Native drag actions unless the official ARC environment exposes a supported drag primitive.
- Automated vulnerability scanning until the experimental harness stabilizes.

## Validation gate

V7 should first pass deterministic fixtures showing that paraphrased goals retain one identity, actionable goals receive executable candidates, cosmetic changes do not count as progress, and productive multi-action policies persist without suppressing necessary replanning. It should also replay the successful Sol `ar25` prefix and verify that neither selector interrupts the validated alignment policy without new contradictory evidence. Only then should classical and QAOA selection be compared on matched runs.
