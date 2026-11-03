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

On the local Results page, the **Run** panel supports both single experiments and sequential batches. Batch mode accepts comma- or space-separated game IDs, one batch ID, one seed, and one action budget. Keep the browser and `run_dashboard.sh` terminal open until the progress counter finishes.

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
.venv/bin/qahs-arc --source arc --game ls20 --condition compare --batch batch-01 --seed 0 --max-steps 4
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
.venv/bin/qahs-arc --source arc --game ls20 --condition compare --batch batch-01 --seed 0 --max-steps 4
```

Classical QAHS only:

```bash
.venv/bin/qahs-arc --source arc --game ls20 --condition qahs_classical --batch batch-01 --seed 0 --max-steps 4
```

## 6. Find the results

- Comparison traces: `output/arc_runs/`
- Official ARC toolkit recordings: `output/arc_recordings/`
- Dashboard replay selector: exported comparisons appear in newest-first order.
- Public dashboard copies: `frontend/public/runs/`

For an initial research check, repeat the same game over several seeds and compare completion, levels completed, actions, model calls, quantum calls, shots, and wall time. Do not treat the local demonstration or a single ARC game as evidence of stronger AI.

## 7. Resume an interrupted batch

Batch progress is checkpointed after every completed game in `output/job_sessions/`. The browser remembers the active job ID. If the dashboard or computer restarts, launch it again with `bash run_dashboard.sh`; the Run panel will restore the job as **Paused** and show a **Resume** button.

Resume restarts only the interrupted game with its original game ID, seed, and action budget. Completed games and their JSON results are retained. It does not restore the middle of an ARC game because the remote environment state cannot be reconstructed safely from a local trace.

OpenAI calls time out after 180 seconds and retry up to three times by default. Complete comparisons have no overall time limit. An optional overall limit can be enabled before starting the dashboard by setting `QAHS_EXPERIMENT_TIMEOUT` to a positive number of seconds:

```bash
export OPENAI_REQUEST_TIMEOUT=180
export OPENAI_MAX_RETRIES=3
export QAHS_EXPERIMENT_TIMEOUT=0
```
# ARC-AGI-2 overnight batch

The local dashboard now has an **ARC-AGI-2 · Luna set** benchmark option. Its checklist contains public-evaluation tasks that GPT-6 Luna Medium solved in the published ARC Prize evaluation. Each task uses one shared Luna hypothesis/output catalog; LLM-only, classical QAHS, and quantum QAHS then select from that same catalog. This keeps the selector comparison to one model request per task.

Choose **Batch**, select the desired task IDs, leave the model at **GPT-6 Luna** and effort at **Medium**, give the batch an ID, and start it. The normal batch checkpoint/resume behavior applies between tasks.

# Alternative-path replay

When an existing ARC-AGI-3 result contains a rejected QAHS physical action, the local result page shows **Alternative replay**. It replays the original action prefix from the same seed, substitutes the selected QAHS action, and optionally follows the recorded policy for three or five actions. It makes no new LLM request. Reports are written under `output/counterfactuals/`.

# V10 archived smoke test

V10 used `v10-success-conditioned-revision`. Its prior smoke procedure used `lp85`, seed 0, and a short budget. Existing V10 runs and the frozen harness remain available for reproduction.

V11 information-gain search is the default. Add `--harness v10` to reproduce the frozen success-conditioned revision harness, or `--harness v7` for the older grounded-goals harness.
