# ARC-AGI-3 Run Guide

## 1. Open the project

```bash
cd "/home/daven/Documents/MSSI/Critical Infrastructure Protection/Final Project/QAHS-Prototype"
```

The `.venv` environment is already installed on this computer. To recreate it later:

```bash
bash setup_arc.sh
```

## 2. Create a key-free test run

```bash
.venv/bin/qahs-arc --source demo --condition compare --max-steps 6
bash run_dashboard.sh
```

Open `http://127.0.0.1:3000`. The Next.js dashboard loads the newest exported run.

Expected demonstration result:

- Without QAHS: success in three actions.
- With QAHS/QAOA: success in two actions.
- Both tabs show grids, model hypotheses, action candidates, QAHS scores, selected actions, and search diagnostics.

This is a harness check created to make the difference visible. It is not research evidence.

## 3. Set credentials for a real run

Create an OpenAI API key for your API project and an ARC-AGI-3 key from `https://three.arcprize.org`. Export them in the terminal that will launch the dashboard:

```bash
export OPENAI_API_KEY='replace-me'
export OPENAI_MODEL='replace-with-a-model-available-to-your-project'
export ARC_API_KEY='replace-me'
```

Do not put real keys in source files, screenshots, or saved traces. The application reads them from environment variables and does not serialize them.

## 4. Run one small ARC comparison

```bash
.venv/bin/qahs-arc --source arc --game ls20 --condition compare --seed 0 --max-steps 4
bash run_dashboard.sh
```

The same game and seed run in three separate sessions. The control follows the language model's recommendation. Exact classical QAHS and simulated-QAOA QAHS use the same persistent hypotheses, candidate scores, and cost function, so their comparison isolates the selection method. Identical state/history inputs reuse one model response across conditions.

## 5. Command-line alternatives

Key-free check:

```bash
.venv/bin/qahs-arc --source demo --condition compare --max-steps 6
```

Real comparison:

```bash
.venv/bin/qahs-arc --source arc --game ls20 --condition compare --seed 0 --max-steps 4
```

Classical QAHS only:

```bash
.venv/bin/qahs-arc --source arc --game ls20 --condition qahs_classical --seed 0 --max-steps 4
```

## 6. Find the results

- Comparison traces: `output/arc_runs/`
- Official ARC toolkit recordings: `output/arc_recordings/`
- Dashboard replay selector: exported comparisons appear in newest-first order.
- Public dashboard copies: `frontend/public/runs/`

For an initial research check, repeat the same game over several seeds and compare completion, levels completed, actions, model calls, quantum calls, shots, and wall time. Do not treat the local demonstration or a single ARC game as evidence of stronger AI.
