# =============================================================================
# DOCKERFILE — AI Proctored Exam Portal
#
# Multi-stage build:
#   Stage 1 (builder) — installs system + Python deps in one layer
#   Stage 2 (runtime) — minimal production image with only what is needed
#
# Base: python:3.11-slim — good balance of size vs compatibility.
#   Alpine is avoided because MediaPipe wheels are not published for
#   musl-based distros and compiling from source adds complexity.
#
# System deps:
#   libgl1-mesa-glx  — provides libGL.so.1 required by OpenCV
#   libglib2.0-0     — GLib runtime, needed by OpenCV's GStreamer backend
#   libsm6, libxext6 — X11 libraries some OpenCV builds link against
#   libxrender-dev   — rendering support for OpenCV highgui (optional but safe)
#   libgomp1         — OpenMP threading support used by NumPy / OpenCV
#
# Railway deployment:
#   The PORT env var is provided by Railway and passed to gunicorn.
#   Setting workers=1 is REQUIRED because detection state lives in
#   process memory (not Redis/DB).
# =============================================================================

# ── Stage 1: builder ───────────────────────────────────────────
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Stage 2: runtime ──────────────────────────────────────────
FROM python:3.11-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN addgroup --system --gid 1001 appgroup && \
    adduser --system --uid 1001 --gid 1001 appuser

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY . .

RUN chown -R appuser:appgroup /app

USER appuser

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:$PORT/proctor/ping')" || exit 1

CMD gunicorn backend:create_app() --bind 0.0.0.0:${PORT:-8080} --workers 1 --timeout 120
