
## 2026-11-04: v10 vc33 (Claude mirror batch)

`20261104T041024Z__v10-success-conditioned-revision__claude-mirror-v10-batch__vc33__...` ended in
`BadRequestError` in all three conditions (LLM-only after 25 actions, both QAHS conditions at
action 1) because the Anthropic account ran out of credit mid-run. It is not valid data. It was
moved out of `frontend/public/runs/` so the dashboard does not list it and the run can be repeated.

## 2026-11-04: v12 lp85 seed 7 (claude-v12-final-lp85-64k)

`...claude-v12-final-lp85-64k__lp85__seed-07__...` ended after 4-7 actions per condition because the
Anthropic account ran out of credit; the trace's `model_settings.call_errors` records the
credit-balance errors. Not valid data; removed from the dashboard so the run can be repeated.
