# ── Stage 1: Build dependencies ──────────────────────────────────
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: Runtime ─────────────────────────────────────────────
FROM python:3.11-slim

# Install ffmpeg and minimal runtime deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

WORKDIR /app

# Copy project files
COPY . .

# Create output directory
RUN mkdir -p /app/output

# Dashboard binds to 0.0.0.0 inside Docker so port mapping works
ENV DASHBOARD_HOST=0.0.0.0
ENV DASHBOARD_PORT=8085

EXPOSE 8085

CMD ["python", "dashboard/server.py"]
