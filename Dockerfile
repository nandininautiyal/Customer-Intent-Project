FROM python:3.10-slim

# Install system dependencies including Node.js for React build
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    curl \
    && curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python dependencies first
# (separate layer so it caches — only rebuilds if requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy frontend and build it
COPY frontend/ ./frontend/
RUN cd frontend && npm install && npm run build

# Copy rest of project
COPY . .

# Create reports directories
RUN mkdir -p reports/artifacts

# Hugging Face Spaces runs as non-root user
# Give permissions to the app directory
RUN chmod -R 777 /app/reports

# Expose port 7860 — HF Spaces requires this exact port
EXPOSE 7860

# Start script
CMD ["bash", "start_hf.sh"]