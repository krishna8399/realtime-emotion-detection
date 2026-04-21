FROM python:3.10-slim

WORKDIR /app

# System dependencies for OpenCV and MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (headless deps, no wandb/kaggle/pytest)
COPY space_requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir "mediapipe==0.10.9"

# Disable Streamlit telemetry prompt
RUN mkdir -p /root/.streamlit && \
    echo '[browser]\ngatherUsageStats = false' > /root/.streamlit/config.toml

# Copy project source (data/ and wandb/ are excluded via .dockerignore)
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Health check — polls the Streamlit health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run Streamlit
ENTRYPOINT ["streamlit", "run", "src/app/app.py", \
    "--server.port=8501", \
    "--server.address=0.0.0.0", \
    "--server.headless=true", \
    "--browser.gatherUsageStats=false"]
