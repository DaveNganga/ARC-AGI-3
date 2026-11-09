# Canonical GPT-6 Luna Run Times

This table summarizes the 21 canonical LP85 runs for seeds 6–26. Each comparison file records three conditions: Luna alone, Classical QAHS, and QAOA QAHS. Times below are the recorded `wall_seconds` values, converted to minutes.

## Shared configuration

- Model: `gpt-6-luna`
- Reasoning effort: `medium`
- Harness: `v12-structured-world-models`
- Game: `lp85`
- Action budget: `40`
- Execution: local QAHS harness calling the remote model service; QAOA used the project’s simulated-QAOA condition, not physical quantum hardware.

## Per-seed wall-clock time

| Seed | Luna alone | Classical QAHS | QAOA QAHS | Published comparison record |
|---:|---:|---:|---:|---|
| 6 | 15.91 min | 24.91 min | 31.41 min | [`seed 6`](./20261103T195204Z__v12-structured-world-models__v12-final-lp85__lp85__seed-06__actions-40__gpt-6-luna__comparison.json) |
| 7 | 18.01 min | 25.47 min | 19.08 min | [`seed 7`](./20261103T205521Z__v12-structured-world-models__v12-final-lp85__lp85__seed-07__actions-40__gpt-6-luna__comparison.json) |
| 8 | 30.19 min | 28.88 min | 43.26 min | [`seed 8`](./20261103T231803Z__v12-structured-world-models__v12-final-lp85__lp85__seed-08__actions-40__gpt-6-luna__comparison.json) |
| 9 | 18.18 min | 37.10 min | 31.00 min | [`seed 9`](./20261104T022151Z__v12-structured-world-models__v12-final-lp85__lp85__seed-09__actions-40__gpt-6-luna__comparison.json) |
| 10 | 15.63 min | 27.15 min | 19.84 min | [`seed 10`](./20261104T035320Z__v12-structured-world-models__v12-final-lp85__lp85__seed-10__actions-40__gpt-6-luna__comparison.json) |
| 11 | 15.00 min | 26.68 min | 15.45 min | [`seed 11`](./20261105T151607Z__v12-structured-world-models__v12-final2-lp85-retry__lp85__seed-11__actions-40__gpt-6-luna__comparison.json) |
| 12 | 14.38 min | 39.74 min | 33.34 min | [`seed 12`](./20261105T164339Z__v12-structured-world-models__v12-final2-lp85-retry__lp85__seed-12__actions-40__gpt-6-luna__comparison.json) |
| 13 | 14.40 min | 29.70 min | 19.85 min | [`seed 13`](./20261105T174747Z__v12-structured-world-models__v12-final2-lp85-retry__lp85__seed-13__actions-40__gpt-6-luna__comparison.json) |
| 14 | 21.50 min | 29.82 min | 28.72 min | [`seed 14`](./20261105T190754Z__v12-structured-world-models__v12-final2-lp85-retry__lp85__seed-14__actions-40__gpt-6-luna__comparison.json) |
| 15 | 20.63 min | 29.83 min | 22.86 min | [`seed 15`](./20261105T202119Z__v12-structured-world-models__v12-final2-lp85-retry__lp85__seed-15__actions-40__gpt-6-luna__comparison.json) |
| 16 | 13.34 min | 20.15 min | 33.31 min | [`seed 16`](./20261105T212812Z__v12-structured-world-models__v12-final2-lp85-retry__lp85__seed-16__actions-40__gpt-6-luna__comparison.json) |
| 17 | 18.01 min | 33.79 min | 27.51 min | [`seed 17`](./20261108T010956Z__v12-structured-world-models__v12-final-gap-s17-24__lp85__seed-17__actions-40__gpt-6-luna__comparison.json) |
| 18 | 48.22 min | 37.73 min | 17.73 min | [`seed 18`](./20261108T025342Z__v12-structured-world-models__v12-final-gap-s17-24__lp85__seed-18__actions-40__gpt-6-luna__comparison.json) |
| 19 | 13.34 min | 37.86 min | 33.13 min | [`seed 19`](./20261108T041812Z__v12-structured-world-models__v12-final-gap-s17-24__lp85__seed-19__actions-40__gpt-6-luna__comparison.json) |
| 20 | 10.19 min | 42.28 min | 21.29 min | [`seed 20`](./20261108T053204Z__v12-structured-world-models__v12-final-gap-s17-24__lp85__seed-20__actions-40__gpt-6-luna__comparison.json) |
| 21 | 34.74 min | 30.83 min | 20.13 min | [`seed 21`](./20261108T065751Z__v12-structured-world-models__v12-final-gap-s17-24__lp85__seed-21__actions-40__gpt-6-luna__comparison.json) |
| 22 | 14.41 min | 43.05 min | 33.24 min | [`seed 22`](./20261108T082843Z__v12-structured-world-models__v12-final-gap-s17-24__lp85__seed-22__actions-40__gpt-6-luna__comparison.json) |
| 23 | 14.94 min | 12.65 min | 33.04 min | [`seed 23`](./20261108T092926Z__v12-structured-world-models__v12-final-gap-s17-24__lp85__seed-23__actions-40__gpt-6-luna__comparison.json) |
| 24 | 18.28 min | 26.78 min | 24.48 min | [`seed 24`](./20261108T103903Z__v12-structured-world-models__v12-final-gap-s17-24__lp85__seed-24__actions-40__gpt-6-luna__comparison.json) |
| 25 | 16.95 min | 39.28 min | 30.29 min | [`seed 25`](./20261106T211958Z__v12-structured-world-models__v12-final3-lp85-s25-34__lp85__seed-25__actions-40__gpt-6-luna__comparison.json) |
| 26 | 15.41 min | 33.14 min | 22.41 min | [`seed 26`](./20261106T223106Z__v12-structured-world-models__v12-final3-lp85-s25-34__lp85__seed-26__actions-40__gpt-6-luna__comparison.json) |

## Aggregate wall-clock time

| Condition | Runs | Total | Mean per run | Range |
|---|---:|---:|---:|---:|
| Luna alone | 21 | 6.69 hr | 19.13 min | 10.19–48.22 min |
| Classical QAHS | 21 | 10.95 hr | 31.28 min | 12.65–43.05 min |
| QAOA QAHS | 21 | 9.36 hr | 26.73 min | 15.45–43.26 min |

The three-condition total is **27.00 recorded wall-clock hours** across 63 condition-runs. This is the sum of individual recorded durations, not necessarily the calendar elapsed time for the complete batch if runs overlapped or were retried.

## Canonical-record selection

- Seed 8 uses the first completed `v12-final-lp85` result.
- Seeds 11–16 use their completed `v12-final2-lp85-retry` records; short incomplete attempts are excluded.
- Seeds 17–24 use `v12-final-gap-s17-24`, and seeds 25–26 use `v12-final3-lp85-s25-34`.
