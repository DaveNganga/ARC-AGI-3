# Group 2 Deliverables — Implemented QAHS System

This document describes the system implemented in the repository. The code is treated as the authoritative definition of the project.

## How ARC input reaches the language model

ARC-AGI-3 returns each observation as structured data containing the game state, available actions, level count, and one or more numeric grid frames. The environment adapter keeps the latest frame as a two-dimensional tuple of integer color values.

The language model receives an exact nearest-neighbor rendering of the grid as a Base64 PNG image. It also receives `frame_summary()`, which converts the numeric grid into text containing:

- grid height and width;
- the inferred background color;
- the count and bounding coordinates of every color; and
- exact horizontal runs of non-background cells using `x,y` coordinates.

The summary is accompanied by a lossless adaptive encoding. Sparse grids use a background value plus every exceptional `x,y:color` cell. Dense grids use the complete row matrix when that form is shorter. The representation therefore preserves every grid cell without always paying the cost of sending a complete matrix.

That text is combined with the game state, available actions, recent evidence, and the persistent belief ledger. It is sent to the OpenAI Responses API as a text prompt. The model's response is constrained by a JSON schema and parsed into hypotheses, predictions, action candidates, or reflection results.

The saved experiment file is JSON and includes the complete numeric grids before and after every action. It records the representation as `image+summary+adaptive-lossless-v1`. The Next.js dashboard reads those JSON files and reconstructs the colored grids for people to view. Luna therefore receives both visual and exact symbolic representations.

---

## 1. Formal hypothesis representation compatible with LLM output

For every turn, the language model returns between eight and sixteen hypotheses. Each hypothesis contains:

- a contiguous integer ID;
- a natural-language statement;
- a confidence value between zero and one; and
- a predicted result for every proposed action candidate.
- an object role, rule, goal condition, and falsification condition.

The harness constructs up to 32 two-step plans from the available controls, visible component centers and boundaries, recent changed cells, and unexplored regions. The model annotates these fixed candidates with predictions, expected progress, and risk rather than inventing the physical action space. QAHS scores the complete plans, executes the first step, observes the result, and replans.

After the selected action is executed, a second model call evaluates the observed result. It returns a persistent belief ledger in which each explanation is marked `supported`, `weakened`, `rejected`, `uncertain`, or `new`. This ledger is included in the next turn so the system can preserve useful explanations and avoid silently restoring rejected ones.

For QAHS selection, `ArcEncoding` converts the hypotheses and action candidates into indexed decision variables. The resulting encoding is shared by the classical and quantum-assisted selectors.

## 2. Search and utility objective

The system scores every proposed action using the hypotheses' predicted outcomes and confidence values. Its score combines:

- information gain: how well the action separates competing explanations;
- expected progress;
- risk;
- novelty relative to prior actions;
- repetition penalties; and
- evidence from earlier actions taken from the same grid state.

The objective favors actions that are expected to reveal which explanation is correct while still rewarding likely game progress. An action that previously produced no useful change from the same frame is penalized, encouraging the agent to perform a different experiment.

Objective components are normalized before encoding. Confidence floors and diversity terms prevent one hypothesis from absorbing the entire search, while level progress and predicted survival receive stronger weight than raw pixel change.

The encoded cost then adds an interaction for every specific hypothesis–action pair. Current confidence is adjusted using the persistent belief ledger: supported explanations gain weight, weakened explanations lose weight, and rejected explanations become unlikely. A pair gains value when its predicted outcome would distinguish a supported hypothesis from its competitors. Missing predictions are penalized. Accumulated evidence therefore changes the classical and quantum search landscape rather than merely breaking ties between action scores.

The three conditions use the same model-produced proposal but select from it differently:

1. **LLM only** follows the candidate recommended by the language model.
2. **Classical QAHS** uses exact classical search over the encoded candidates and their objective values.
3. **Quantum QAHS** uses the QAOA search backend to select from the same encoded decision problem.

## 3. Working QAHS implementation

The implemented reasoning loop is:

```text
ARC observation
→ numeric grid converted to coordinate-based text
→ LLM proposes hypotheses and candidate actions
→ action scores are calculated
→ LLM, classical QAHS, or quantum QAHS selects an action
→ ARC executes the action
→ LLM reflects on the result
→ belief ledger is updated
→ next turn
```

Each comparison starts the same ARC game three times using the same seed and action budget. This produces an LLM-only episode, a classical-QAHS episode, and a quantum-QAHS episode. Every condition receives the same initial challenge, while later observations reflect the different actions selected by that condition.

The system supports individual experiments and sequential batches. Each saved comparison records its game, seed, action budget, model, batch ID, harness version, episode traces, selector diagnostics, timing, model-call count, quantum-call count, and shot count.

## 4. Quantum algorithm and its role

The quantum-assisted condition uses the Quantum Approximate Optimization Algorithm (QAOA). The hypotheses and candidate actions are encoded into a finite binary decision space. Each basis-state cost contains the shared action utility plus that hypothesis–action pair's evidence support and information contribution. QAOA searches this history-aware landscape for a low-cost combination.

The implemented simulator uses:

- QAOA depth of two;
- three optimizer restarts;
- up to 80 optimizer iterations;
- 512 measurement shots per selection; and
- a deterministic seed derived from the game seed and turn number.

The measured quantum result identifies the action candidate selected for the environment. The quantum component therefore acts as an action selector inside a larger classical and language-model-controlled reasoning loop.

## 5. Resource and scaling measurements

Every quantum-assisted turn records the number of decision qubits, simulator calls, and measurement shots. With 512 shots per QAOA selection, the total shot count grows with the number of actions taken in the episode.

The repository also includes scaling utilities that estimate statevector memory using:

\[
2^n \times 16\text{ bytes}
\]

for `n` complex128 qubits. This connects the size of an encoded decision problem to the memory required by the local quantum simulator.

The dashboard separates proposal, reflection, ARC environment, objective, and selector time. QAOA records the exact classical minimum, selected cost, regret, selected rank, and probability assigned to the exact optimum. This distinguishes model latency from search quality and simulator cost.

## 6. Correctness testing

The repository tests the system at several levels:

- parsing and validation of schema-constrained model responses;
- preservation of grid dimensions and coordinates in `frame_summary()`;
- action validation, including click-coordinate bounds;
- persistent reflection and belief-ledger handling;
- action scoring and repeated-action penalties;
- classical and QAOA selector behavior;
- comparison-trace construction;
- saved-run naming and metadata; and
- the deterministic local demonstration environment.

The local demonstration provides a key-free, repeatable test of the full comparison harness before the live ARC service and OpenAI model are used.

## 7. Controlled experimental comparison

Each comparison holds the following setup values constant across all three conditions:

- ARC game ID;
- game seed;
- maximum actions;
- OpenAI model;
- prompt and structured-output schemas;
- objective calculation; and
- initial environment state.

A caching reasoner reuses a model result when two conditions encounter exactly the same observation and history. Once their selected actions differ, each condition follows its own valid trajectory and receives observations produced by that trajectory.

The batch interface runs a list of games sequentially under one batch ID. Run filenames and JSON metadata preserve the harness version, batch, game, seed, action budget, model, and run type. The dashboard can then compare one run or aggregate a version-consistent batch by a selected metric.

## 8. Experimental results produced by the system

For each condition, the system records:

- whether the game was solved;
- levels completed;
- actions taken;
- useful actions;
- changed cells;
- model calls;
- quantum calls and shots;
- wall-clock time;
- selected candidates and their scores;
- hypotheses, predictions, and recommendations;
- before-and-after observations; and
- post-action reflections and belief updates.

The Next.js results dashboard presents these records as direct comparisons among LLM only, classical QAHS, and quantum QAHS. Users can switch between individual runs and batch summaries, select a comparison metric, inspect run configuration, and open the detailed turn trace when deeper analysis is needed.

Together, these outputs support the project's central experiment: determine whether explicitly maintaining competing hypotheses and using classical or quantum-assisted search changes the quality and efficiency of an LLM's actions in an unfamiliar interactive reasoning task.
