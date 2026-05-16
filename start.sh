#!/bin/bash

echo "=== Customer Intent Engine — Render Startup ==="

# Install Node and build React frontend
echo "Installing Node.js..."
curl -fsSL https://deb.nodesource.com/setup_18.x | bash -
apt-get install -y nodejs

echo "Building React frontend..."
cd frontend
npm install
npm run build
cd ..

echo "Build complete. Static files in frontend/dist/"


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