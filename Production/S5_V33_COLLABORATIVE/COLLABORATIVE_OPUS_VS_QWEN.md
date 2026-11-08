# Collaborative condition: Claude Opus 5.5 vs Qwen3:8B

Paired comparison of the two V3.3 `collaborative` runs. Machine-readable form:
`s5_16m_collaborative_opus_vs_qwen.json`.

## Why these two are comparable

Both runs share `benchmark_id`, `manifest_sha256` (`59e6e3b2…`), `metric_spec_sha256`
(`d9d0afba…`), `profile: paper`, `max_steps: 12`, the 16,777,216-world hypothesis space, and the
same per-episode seeds, so they played the same 18 episodes. The S5 prompt is text-only for every
provider, so unlike the ARC-AGI-3 v11/v12 path this comparison is **modality-matched** — neither
model received anything the other did not.

Differences that remain, and are not controlled: Opus ran at `ANTHROPIC_EFFORT=medium` and cannot
set temperature; Qwen ran at `temperature=0` with thinking disabled. Effort is therefore not a
controlled variable.

## Scores

| Domain | Opus 5.5 | Qwen3:8B | Δ (qwen − opus) | n | qwen better | opus better | ties | p |
|---|---|---|---|---|---|---|---|---|
| G  generalization | 0.000 | 0.000 | +0.000 | 5 | 0 | 0 | 5 | 1.000 |
| A  adaptation | 1.000 | 1.000 | +0.000 | 3 | 0 | 0 | 3 | 1.000 |
| Au autonomy | 0.708 | 0.708 | +0.000 | 5 | 0 | 0 | 5 | 1.000 |
| T  transfer | 0.444 | 0.500 | **+0.056** | 3 | 1 | 0 | 2 | 1.000 |
| R  robustness | 0.000 | 0.000 | +0.000 | 3 | 0 | 0 | 3 | 1.000 |
| **SAI** | **0.4304** | **0.4416** | **+0.0111** | | | | | |

Two-sided exact sign test, paired per episode. **No domain reaches significance**; every one is
p = 1.0.

G, A, Au and R are identical **per episode**, not merely equal on average:

```
Au  opus = [0.8333, 0.7778, 0.5111, 0.4167, 1.0]
    qwen = [0.8333, 0.7778, 0.5111, 0.4167, 1.0]
```

The entire SAI difference comes from **one transfer pair**:

```
T   opus = [0.500, 0.500, 0.333]
    qwen = [0.500, 0.500, 0.500]      <- T03 is the only disagreement in the whole benchmark
```

## The finding: QAHS sets the behaviour, not the model

The identical scores are not identical trajectories. **Action sequences differ in 15 of 18
episodes**, often from turn 1:

| | Opus | Qwen |
|---|---|---|
| Identical action sequence | 3 / 18 episodes (`G03`, `G03_perturbed`, `T03_cold`) | |
| First divergence | turn 1 in 5 episodes; turn 12 at the latest | |

Yet the three behavioural components that make up autonomy —
`novel_experiment_rate`, `redundant_action_rate`, `no_change_action_persistence` — are **identical
per episode for all five generalization episodes** (the ones Au is computed from), and identical in
13 of 18 episodes overall. Two different action sequences with the same novelty, redundancy and
no-change statistics.

The arbitration tallies explain it:

| Decision | Opus | Qwen |
|---|---|---|
| `qa_hs` (QAHS action taken) | 189 (92.6%) | 186 (93.9%) |
| `diagnostic` | 11 (5.4%) | 9 (4.5%) |
| `agreement` | 4 (2.0%) | 3 (1.5%) |
| `llm` | **0** | **0** |
| `llm_action` | **0** | **0** |
| Total turns | 204 | 198 |

**Neither model ever had its action chosen.** In the collaborative condition the LLM is a hint fed
into `choose_action`, and the trajectory is produced by the QAHS planner, which is deterministically
seeded from the shared manifest. So the behavioural profile — and with it four of the five domain
scores — is a property of the planner, and swapping a frontier model for a local 8B barely moves it.

This is also why it is the more informative result for Qwen specifically:
`CLAUDE_RUN_NOTES_V33.md` records Qwen's hypothesis parsing as complete on ~90% of turns against
Opus's 63%, so the LLM branch was *available* far more often for Qwen and still never won. The
composite puts 78% of its weight on log posterior, which the LLM's single guess rarely beats
against the posterior MAP world.

## What this does not say

- **Not a capability ranking.** It is not evidence that Qwen3:8B matches Opus 5.5 at reasoning. It
  shows that this condition rarely lets either model act, so the condition cannot distinguish them.
  The LLM-only rows are where the models separate: Opus 0.718 vs Qwen 0.400.
- **Not significant.** 18 episodes per condition, every domain p = 1.0, one differing episode.
  One solve moves G by 0.2.
- **Not reproducible on the Qwen side.** The adapter sends no `seed`, so `temperature=0` alone does
  not give bitwise reproducibility; measured non-determinism on this GPU. Opus cannot set
  temperature at all.
- **No quantum-advantage claim.** The six 4-qubit QAOA registers are uncoupled (36 intra-register
  `rzz`, no cross-family term), so the cost function is classically exact.
- G = 0.0 and R = 0.0 for both models: neither solved any generalization or robustness episode.

## Full three-condition context

| Condition | Opus 5.5 SAI | Qwen3:8B SAI |
|---|---|---|
| LLM-only | 0.718 | 0.400 |
| QAHS | 0.459 | 0.475 |
| Collaborative | 0.430 | 0.442 |

For both models collaborative lands below QAHS, and for both the arbiter never chose the LLM.
