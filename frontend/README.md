# QAHS Next.js dashboard

This is the only QAHS dashboard. Python remains the experiment engine; this frontend reads sanitized copies of completed runs. A localhost-only panel calls the Python engine through `scripts/local_run_server.py`.

## Local use

From the project root:

```bash
bash run_dashboard.sh
```

This refreshes `public/runs/`, starts the local experiment service on `127.0.0.1:8765`, and starts the site at `http://127.0.0.1:3000`. The Run panel is shown only on localhost.

## Publish on Vercel

1. Commit this `frontend/` directory, including the sanitized files in `public/runs/`.
2. Import the repository in Vercel.
3. Set **Root Directory** to `frontend`.
4. Keep the detected Next.js build settings and deploy.

The deployed site is display-only. Run experiments locally, rerun `scripts/export_dashboard_runs.py`, then commit and push the updated public runs.
