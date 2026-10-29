# Backlog

Plans that have been designed but not started. Pick these up after the current roadmap phase in [`IMPROVEMENTS.md`](IMPROVEMENTS.md).

## B1. Token-efficient grid encoding

What the ARC docs say:
- **No image or OCR step is needed.** ARC-AGI-3 already returns frames as JSON integer grids: up to 64×64, values 0–15, origin top-left, (x, y) order (docs.arcprize.org/game-schema).
- `ArcSDKEnvironment._last_grid` already turns these frames into ints, and `frame_summary` sends text to the LLM. The LLM never sees an image.

Planned work:
- **Measure first:** record `response.usage` input/output tokens for each proposal and reflection in the trace, and show them in the dashboard.
- **Selectable encodings** (new `frame_encoding.py`):
  - `runs`: the current default;
  - `objects`: connected components with color, bounding box, size, and shape;
  - `hex`: exact grid, one character per cell;
  - `diff`: only the changed cells, for the reflection prompt.
- New CLI option `--frame-format`, logged in `model_settings`.
- Replace the Python `repr` history dumps in prompts with compact JSON.
- **Offline mode:** `Arcade(operation_mode=OperationMode.OFFLINE)` reads `environment_files/`. It needs no key, has no rate limit, runs at about 2,000 FPS, and suits development and token experiments. Add a `--arc-mode offline` option.

## B2. Session management and safe resume

What the ARC docs say:
- The toolkit's `reset()` and `step()` **return `None` on failure** instead of raising an error.
- Online sessions depend on a server `guid` plus routing cookies.
- Scorecards auto-close after 15 minutes. The online rate limit is 600 requests per minute.
- There is no read-only "get state" call, and re-sending a lost step may apply the action twice.

Planned work:
- **Journal:** one append-only `output/arc_sessions/<run_id>/journal.jsonl` per run, with fsync after each event. It records run_started, proposal, action_committed, reflection, episode_finished, connection_lost/resumed, and run_finished.
- **Failure classification:** transient (connection error, timeout, HTTP 429/5xx, toolkit `None`) vs. fatal (validation errors, 401/403).
- **Detect and wait:** probe with exponential backoff (ARC `/api/healthcheck`, or a cheap OpenAI call) up to a maximum wait.
- **Safe restore by deterministic replay:**
  - start a fresh environment with the same game and seed;
  - replay only committed actions;
  - check each frame hash against the journal;
  - stop loudly on any mismatch (`replay_divergence`);
  - never assume a lost step was applied.
- **No repeated LLM calls:** journaled proposals and reflections are reused on resume, and the shared cache is pre-filled from the journal.
- **Entry points:** `qahs-arc --resume <run_id|latest>`, a warning on start about unfinished runs, and a local runner `POST /resume` with a dashboard "Resume" button.
- **Final trace:** written only at run_finished. It records `resumed`, `connection_events`, and `replayed_actions`; replay steps are not counted as agent decisions.
- **Caveat:** in online mode, replay sends real actions to the server, so the scorecard shows more actions than the agent decided. Use offline mode for development.
- **Tests:** fake environment that drops on step 3, then resumes; replay divergence; resume after the process is killed, with the final trace identical to an uninterrupted run.
