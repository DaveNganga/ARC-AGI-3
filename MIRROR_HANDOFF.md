# Handoff: run the Claude mirror study (cloud agent)

**Scope: this task only.** Run the pre-registered matrix, push the results, and report. Do not change code, do not start other work, and do not touch any branch other than this one.

## What this is

This branch (`mirror/claude-on-gpt-harness`) is the exact harness (commit `b57e381`) that produced 9 published GPT (`gpt-6-luna`) ARC-AGI-3 comparisons. Only a Claude provider has been added. The study re-runs those 9 comparisons with `claude-opus-5-5`, so that **the model is the only difference**. That is why the code must stay as it is: any edit breaks the comparison.

Each comparison plays one ARC-AGI-3 game online under three conditions (LLM-only, classical QAHS, QAOA QAHS). It makes one model call per turn.

## Preconditions (check, don't print values)

```bash
for v in ARC_API_KEY ANTHROPIC_API_KEY; do [ -n "${!v:-}" ] && echo "$v set" || echo "$v MISSING"; done
echo "ANTHROPIC_EFFORT=${ANTHROPIC_EFFORT:-unset (script defaults to medium)}"
git branch --show-current          # expect mirror/claude-on-gpt-harness
.venv/bin/python -c "import anthropic, arc_agi, numpy, scipy; print('deps ok')"
```

If `.venv` is missing, run:
```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.lock && .venv/bin/pip install -e . --no-deps && .venv/bin/pip install 'anthropic>=1.0'
```

**Stop and report if a key is missing.** Never echo, log, or commit key values.

## Run

```bash
bash scripts/run_mirror_matrix.sh
```

What the script does, in order:

- It runs these 9 runs: ar25/0/20, vc33/0/30, lp85/0/30, r11l/0/20, s5i5/0/20, vc33/1/30, ar25/1/20, r11l/1/20, s5i5/1/20 (game/seed/action budget).
- After each run, it exports a sanitized trace to `frontend/public/runs/`, writes a log to `mirror_logs/`, commits, and pushes to `origin`.
- It **skips** runs that already have a `claude-opus-5-5` trace, so it is safe to re-run after an interruption.
- At the end, it writes `mirror_logs/compare.md` (Claude vs GPT table) and commits it.

Expect about 550 model calls and roughly 4–8 hours. Each run takes 20–60 minutes, with no output in between. That is normal.

## If something goes wrong

| Situation | Action |
|---|---|
| Script prints `ABORT: … produced no trace` | Read the log it names. If it is an outage (connection error, 5xx, 429, ARC unavailable), wait about 10 min and re-run the script **once**; it resumes. If it is auth, a missing package, or anything else, stop and report. Do not fix code. |
| A run finishes but an episode shows an `error` or 0 actions | This is valid data under the pre-registered rules. Do not re-run it; keep going. |
| `push failed` / `could not push` | Results stay committed locally. If the session can only push to its own branch, push there (`git push origin HEAD:<allowed-branch>`) and report that branch name. |
| Session is about to time out | Nothing extra to do: every finished run is already pushed. Report which runs completed. |
| The same run fails twice from an outage | Stop and report; the user decides. |

Do **not**:
- edit any file other than what the script writes;
- rebase, force-push, or merge;
- change `ANTHROPIC_MODEL` or `ANTHROPIC_EFFORT`;
- re-run a run that completed;
- delete traces or logs, even for failed runs.

## Done when

- `mirror_logs/compare.md` exists and is pushed, or you have stopped under one of the rules above.
- `git status` is clean, and `git log origin/<branch>` shows the "Mirror run: …" commits.

## Report back (this exact shape)

1. **Branch pushed to:** name and last commit hash.
2. **Runs completed:** list of game/seed/budget, plus any that crashed or were skipped, with a one-line reason each.
3. **Re-runs:** which runs were re-run after an outage, if any, and why.
4. **Comparison:** paste `mirror_logs/compare.md` verbatim. Note whether it printed a WARNING about mixed harness versions; it should not.
5. **Anything unexpected**, quoting log lines (with key values redacted).

Do not interpret the results or draw conclusions; the user's local session does the analysis.
