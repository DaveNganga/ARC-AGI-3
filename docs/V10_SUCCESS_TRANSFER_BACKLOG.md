# Harness v10: Success Transfer Backlog

**Harness name:** `v10-success-conditioned-revision`  
**Status:** Implemented as the default ARC-AGI-3 comparison harness.

## Motivation

In the v9 `lp85` batch, the LLM completed level 1 after five clicks on `e3`, but did not form a durable explanation of why the sequence worked. On level 2 it reverted to local experimentation, clicked `e6` 43 times and `e7` 16 times, and completed no additional levels. The current harness records progress numerically but does not give the model a dedicated opportunity to compare the successful transition, generalize it, and map that rule onto the next layout.

## 1. Success-condition consolidation

Trigger one dedicated consolidation step whenever `levels_completed` increases or the game reaches `WIN`.

Provide the reasoner with:

- The frame at the beginning of the successful action episode.
- The frame immediately before the level transition.
- The transition animation and first frame of the new level.
- The ordered action sequence associated with the successful episode.
- Stable object attributes including color, size, shape, bounds and relative position—not only level-local entity IDs.
- Measured changes from every action, clearly distinguishing the terminal action from earlier prerequisites.

Ask for several competing transferable explanations rather than one accepted story. Each explanation must include:

- The proposed causal rule.
- Which actions were necessary, incidental or still uncertain.
- The visible success condition.
- The invariant attributes expected to transfer across levels.
- A mapped analogue in the new level.
- One executable test, its predicted result and a falsification condition.
- Confidence supported by the recorded transition rather than visual plausibility alone.

The implementation audits the hypothesis active at the successful transition. It records supported and contradicted semantic components, distinguishes an incorrect or coincidental explanation from a supported one, and uses the corrected relational lesson in subsequent Luna vocabulary calls. Success no longer automatically awards full evidence to every component of the selected hypothesis.

## Persistence and QAHS integration

- Save consolidated rules in a game-level ledger that persists for the complete episode instead of the six-turn rolling history.
- Keep the original evidence frames, action sequence and model text in the trace for auditability.
- Convert each transferable explanation into canonical semantic atoms and an executable action or short policy.
- Let classical and quantum QAHS select among the competing transfer hypotheses using the same objective.
- Give a successful policy one controlled transfer attempt on the next level; do not assume identical coordinates or entity IDs.
- Preserve failed transfer evidence so the same analogy is not repeatedly retried without a meaningful change.

## Safeguards

- Do not equate a final action with the complete cause when several earlier actions changed the state.
- Do not treat pixel change alone as success; the level counter or `WIN` state remains authoritative.
- Separate reusable rules from layout-specific coordinates.
- Retain multiple explanations when the successful trajectory is causally ambiguous.
- Limit consolidation to one additional model call per level transition unless contradictory evidence justifies another review.

## Validation gate

1. Replay the successful `lp85` level-1 trace without making new ARC or model calls.
2. Verify that the consolidation input contains the five-click sequence and the before/success/new-level frames.
3. Confirm that at least one retained hypothesis describes a transferable color, object-state or repeated-interaction rule rather than only “click `e3`.”
4. Confirm that proposed level-2 tests reference current attributes and executable coordinates rather than stale entity IDs.
5. Verify that a failed transfer is recorded and does not loop indefinitely.
6. Compare v9 and v10 on matched `lp85` seeds before adding v10 to a larger batch.

## Additional v10 candidates

Reserved for further findings from the v9 batch. No additional items are approved yet.
