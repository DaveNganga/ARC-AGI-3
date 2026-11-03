# Experiment Metrics

The primary outcomes are reported separately and in this order:

1. levels completed or game solved;
2. actions required to reach that progress;
3. normalized search regret;
4. uncached model calls required to reach that progress; and
5. useful-action percentage as supporting evidence.

Wall-clock time is descriptive rather than a primary outcome because API and network latency vary independently of decision quality.

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
