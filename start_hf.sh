#!/bin/bash

echo "=== Customer Intent Engine — Hugging Face Startup ==="

# Run pipeline only if artifacts don't exist
if [ ! -f "reports/artifacts/ensemble_meta.pkl" ]; then
    echo "No artifacts found. Running training pipeline..."
    python orchestrator.py
else
    echo "Artifacts found. Skipping training."
fi

echo "Starting FastAPI server on port 7860..."
gunicorn api.main:app \
    --workers 1 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:7860 \
    --timeout 180