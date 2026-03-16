# ── Stage 1: Builder ──────────────────────────────────────────────────────────
FROM python:3.10.17-slim AS builder

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && apt-get upgrade -y && rm -rf /var/lib/apt/lists/* \
 && pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt


# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM python:3.10.17-slim AS runtime

LABEL maintainer="DPE ML Team"
LABEL description="EDT Prediction API for Japan Pizza Delivery"
LABEL version="1.0.0"

RUN apt-get update && apt-get upgrade -y && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed packages from builder — MUST use same base image as builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code and model
COPY src/      ./src/
COPY api/      ./api/
COPY outputs/  ./outputs/

# Non-root user for security
RUN useradd -m -u 1001 appuser && chown -R appuser:appuser /app
USER appuser

# Health check — stdlib only, no extra dependency
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health').read()" || exit 1

EXPOSE 8000

ENV MODEL_PATH=outputs/edt_model.pkl
ENV PYTHONPATH=/app

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
