#!/bin/bash

echo "=== Customer Intent Engine — Hugging Face Startup ==="

echo "Current directory: $(pwd)"
echo "Contents of reports/artifacts:"
ls -la reports/artifacts/ 2>/dev/null || echo "Directory not found"

if [ -f "reports/artifacts/ensemble_meta.pkl" ]; then
    echo "Artifacts found. Skipping training."
else
    echo "No artifacts found. Checking dataset..."
    if [ -f "data/raw/online_shoppers_intention.csv" ]; then
        echo "Dataset found. Running training pipeline..."
        python orchestrator.py
    else
        echo "WARNING: Dataset not found. API will start without models."
    fi
fi

echo "Starting FastAPI server on port 7860..."
gunicorn api.main:app \
    --workers 1 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:7860 \
    --timeout 180