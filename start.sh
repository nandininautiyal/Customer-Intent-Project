#!/bin/bash

echo "=== Customer Intent Engine — Render Startup ==="


if [ ! -f "reports/artifacts/ensemble_meta.pkl" ]; then
    echo "No artifacts found. Running training pipeline..."
    python orchestrator.py
else
    echo "Artifacts found. Skipping training."
fi

echo "Starting FastAPI server..."
gunicorn api.main:app \
    --workers 1 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:$PORT \
    --timeout 120