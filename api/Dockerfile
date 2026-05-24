# Pit Wall Intelligence API -- production-shaped container.
#
# Build:
#   docker build -t pitwall-api -f api/Dockerfile .
# Run:
#   docker run -p 8000:8000 -v $(pwd)/data/processed:/app/data/processed pitwall-api
# Healthcheck:
#   curl http://localhost:8000/health
#
# The container mounts data/processed/ for the joblib models. In production
# you'd publish those to S3/GCS and pull at start-time. For portfolio scale
# the volume mount is enough.

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_PYTHON=python3.11

WORKDIR /app

# Install uv for fast deps
RUN pip install --no-cache-dir uv==0.4.30

# Layer 1: copy lock + manifest only -- maximises Docker cache hits
COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/

# Install only runtime deps (no dev extras) to keep the image lean
RUN uv pip install --system --no-cache .

# Layer 2: copy the API code
COPY api/ ./api/

# Models live here; mounted at run-time
RUN mkdir -p /app/data/processed

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request,sys; urllib.request.urlopen('http://localhost:8000/health',timeout=3); sys.exit(0)" || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
