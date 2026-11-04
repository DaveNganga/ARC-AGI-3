# Experiment Metrics

The primary outcomes are reported separately and in this order:

1. levels completed or game solved;
2. actions required to reach that progress;
3. normalized search regret;
4. uncached model calls required to reach that progress; and
5. useful-action percentage as supporting evidence.

Wall-clock time is descriptive rather than a primary outcome because API and network latency vary independently of decision quality.

## Dashboard run metrics

The four bars in a single-run result use trace-auditable definitions. They do not treat all visible motion as success.

- **Progress** is the number of levels reported complete by the game. A changed frame without a level transition is not progress.
- **Efficiency** is executed actions divided by completed levels. Lower is better. It is reported as unavailable when no level is completed; assigning an unsolved run a numeric efficiency score would imply evidence that does not exist.
- **Exploration** is action-space coverage: distinct physical actions attempted divided by the distinct physical actions offered anywhere in the saved trace. An action includes its type, source coordinates, and destination coordinates when present. This measures breadth, not quality, so a larger value is not automatically better.
- **Useful actions** is the percentage of executed actions that produced objective evidence. Evidence means game progress or terminal feedback, a classified physical outcome other than `NO_CHANGE`, or a material recorded belief update (posterior total-variation distance of at least 0.05, explicit validation/refutation, or hypothesis retirement). A frame difference is used only as a fallback for old traces that predate classified outcomes. This metric describes information yield; it does not claim that every state change moved toward the unknown game objective.

Repeated actions are not automatically penalized. A repeated action remains useful while it continues to produce objective evidence. It becomes an ineffective repeat only when both the previous and current identical actions produce no such evidence.

## Stronger-AI capability proxy scores

The run-level table compares LLM-only and quantum QAHS on a numeric zero-to-one scale. Classical remains in the ordinary charts as a selector baseline. These are trace-derived proxies for the five definitions in `Teachers version/QAHS_Defined.pptx`, not the professor's controlled multi-episode benchmark scores.

- **Generalization (G):** completed levels divided by the game-reported levels required to win. If the required level count was not recorded, only a solved episode receives 1.000. This becomes held-out generalization evidence only when the game was excluded from harness development.
- **Adaptation (A):** after each observed level transition, `max(0, 1 - recovery latency / episode action count)`, averaged across transitions. Recovery means the first later action that yields objective evidence. An episode with no level transition receives 0.000. This is a natural level-transition proxy rather than an injected hidden-rule test.
- **Autonomy (Au):** the equal-weight mean of (1) distinct physical actions divided by executed actions, (2) avoidance of repeating the same action immediately after `NO_CHANGE`, and (3) objective environmental-evidence yield. Environmental evidence means a classified physical change, level progress, terminal feedback, or a legacy frame change; an agent's belief update alone does not earn autonomy credit. Productive persistence can still earn evidence credit even though repeated actions do not increase the novelty component.
- **Transfer (T):** 0.000 because a single saved run has no matched warm/cold related-world pair.
- **Robustness (R):** 0.000 because a single saved run has no matched clean/perturbed pair.

The displayed **SAI proxy** is the equal-weight mean `(G + A + Au + T + R) / 5`. Assigning zero to T and R is a presentation convention requested for this project; it means those capabilities were not demonstrated by the run, not that a controlled experiment proved the system has zero transfer or robustness.

## Exploratory combined score

The dashboard also reports a within-run exploratory score:

\[
100(0.50P + 0.20A + 0.20S + 0.10C)
\]

where:

- `P` is levels completed relative to the best condition in that matched comparison;
- `A` is levels per action relative to the best condition;
- `S` is one minus mean regret normalized by the largest mean regret;
- `C` is levels per uncached model call relative to the best condition.

All components range from zero to one. When no condition completes a level, progress, action efficiency, and call efficiency are zero; the score then reflects search quality only. When every condition has zero regret, every condition receives full search-quality credit.

The combined score is exploratory and must not replace the component metrics. Its weights were selected before the planned `v4` study: 50% progress, 20% action efficiency, 20% search quality, and 10% model-call efficiency. Results should therefore show both the combined score and its four components.
