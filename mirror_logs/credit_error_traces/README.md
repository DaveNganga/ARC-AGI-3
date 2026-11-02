# Seed-1 traces lost to an API billing error

These four Claude traces (vc33/1/30, ar25/1/20, r11l/1/20, s5i5/1/20) were recorded on
2026-11-02 around 06:34 UTC, after the Anthropic account ran out of credit. Every episode
stopped with `BadRequestError`; a direct API call at that time returned
"Your credit balance is too low to access the Anthropic API".

They are kept here for the record and moved out of `frontend/public/runs/` so that
`scripts/run_mirror_matrix.sh` re-plays these four runs once credit was restored. The
original logs remain in `mirror_logs/`.
