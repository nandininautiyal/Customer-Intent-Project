FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    curl \
    && curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install huggingface_hub separately to download artifacts
RUN pip install huggingface_hub hf-xet

COPY frontend/ ./frontend/
RUN cd frontend && npm install && npm run build

COPY . .

RUN mkdir -p reports/artifacts

# Download pre-trained artifacts from HF at build time
RUN python - <<'EOF'
from huggingface_hub import hf_hub_download
import os

repo_id = "nandininautiyal/customer-intent-engine"
artifacts = [
    "reports/artifacts/scaler.pkl",
    "reports/artifacts/logistic.pkl",
    "reports/artifacts/xgboost.pkl",
    "reports/artifacts/neural.pth",
    "reports/artifacts/ensemble_meta.pkl",
    "reports/artifacts/bandit.pkl",
    "reports/artifacts/visitor_type_encoder.pkl",
]

os.makedirs("/app/reports/artifacts", exist_ok=True)

for artifact_path in artifacts:
    try:
        print(f"Downloading {artifact_path}...")
        hf_hub_download(
            repo_id=repo_id,
            filename=artifact_path,
            repo_type="space",
            local_dir="/app",
            local_dir_use_symlinks=False,
        )
        print(f"Done: {artifact_path}")
    except Exception as e:
        print(f"Skipping {artifact_path}: {e}")

print("Artifact download complete.")
EOF

RUN chmod -R 777 /app/reports

EXPOSE 7860

CMD ["bash", "start_hf.sh"]