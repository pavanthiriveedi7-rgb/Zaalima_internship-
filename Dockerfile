# ── Base Image ──────────────────────────────────────────────────────────
FROM python:3.10-slim

# ── System Dependencies ─────────────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Working Directory ───────────────────────────────────────────────────
WORKDIR /app

# ── Python Dependencies ─────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application Code ────────────────────────────────────────────────────
COPY . .

# ── Environment Variables ───────────────────────────────────────────────
ENV QUERY_TOWER_PATH=/app/models/query_tower
ENV REDIS_HOST=redis
ENV REDIS_PORT=6379
ENV FAISS_INDEX_PATH=/app/models/faiss_index

# ── Expose API Port ─────────────────────────────────────────────────────
EXPOSE 8000

# ── Health Check ────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ── Entrypoint ──────────────────────────────────────────────────────────
CMD ["uvicorn", "week4.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
