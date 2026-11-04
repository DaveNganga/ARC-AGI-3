# Harness V11 — Information-gain search

V10 is frozen as `v10-success-conditioned-revision`. V11 is a separate harness named
`v11-information-gain-search`; old runs remain reproducible with `--harness v10`.

V11 replaces the growing semantic patch stack with one evidence loop:

1. Luna proposes 6–10 complete, falsifiable hypotheses.
2. Every hypothesis predicts the outcome of every available physical action.
3. Python classifies the observed result into a fixed outcome vocabulary.
4. Python updates hypothesis probabilities from prediction accuracy.
5. Classical exact search and QAOA minimize the same three-action plan cost.
6. Only the first action is executed, then the system observes and replans.

The objective combines expected information gain, expected game progress, action risk,
and recent repetition. Success transitions are shown to Luna as before/after states on
the next hypothesis refresh, but success only supports a hypothesis when its prediction
matches the measured result.

Each trace records the full prediction matrix, probabilities before and after evidence,
the selected ordered plan, the objective components, and the objective outcome.

The post-smoke refinement uses global entity matching, removes only unambiguous full-scene
background regions while retaining ambiguous and edge-attached components as testable objects, forces a new choice after
two identical no-change results, weights disagreements involving progress more strongly, and
values later plan steps only for distinctions that earlier steps leave unresolved.

Hypotheses may now own a test procedure with a machine-checkable continuation contract: the
repeated action, tracked entity, optional target entity, predicted relationship change, and explicit
confirmation and refutation outcomes. There is no arbitrary repetition horizon. A procedure continues
while its predicted outcome and relationship change remain true, and stops only when its own evidence
contract confirms or refutes it. A better-supported updated hypothesis may inherit and extend the
active procedure when it explains the same observed action more convincingly.

The validated-persistence refinement separates the clicked actuator from the entities it affects. Three or more
coordinated entity movements are classified as `GROUP_TRANSFORM` rather than a single-object `MOVE`. A confirming
outcome now strengthens and continues a persistent procedure; it does not end that procedure. The claimed entity
relationship is checked before confirmation is accepted, preventing a generic board movement from confirming an
incorrect statement such as “the clicked control moved toward the target.”
After confirmation, the persistent action takes precedence over unused exploratory steps in the original test
sequence. This prevents a confirmed control action from being followed by unrelated probes before it can repeat.

Physical and game effects are now recorded separately. A level transition can therefore be represented as
`physical_outcome=GROUP_TRANSFORM` together with `game_outcome=PROGRESS`. When an active procedure correctly
predicts the physical effect and produces progress, its hypothesis is recorded as `validated_level` for the completed
level rather than refuted because the primary outcome label changed. The validated rule is included in the next
level's success audit as transferable but still fallible evidence.
Two high-confidence contradictions and a posterior below 2.5% retire a hypothesis for the current
level. Retired statements remain in the trace and are supplied to Luna to prevent immediate
rephrasing; a new level clears retirement because its rules or controls may differ.

## Smoke test

Use `lp85`, seed `0`, and `12` actions. Set the batch ID to `v11-smoke-s0`.
