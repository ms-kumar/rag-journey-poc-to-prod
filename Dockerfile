# Multi-stage Dockerfile for RAG API
# Stage 1: Build stage with uv
# Stage 2: Production runtime

# Pin versions for reproducibility
ARG PYTHON_VERSION=3.11
ARG UV_VERSION=0.5

# ============================================
# UV binary stage (workaround for --from variable expansion)
# ============================================
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

# ============================================
# Stage 1: Builder
# ============================================
FROM python:${PYTHON_VERSION}-slim AS builder

# Install uv (pinned version for reproducibility)
COPY --from=uv /uv /usr/local/bin/uv

# Set working directory
WORKDIR /app

# Copy dependency files first for better caching
# README.md is needed because pyproject.toml references it
COPY pyproject.toml uv.lock README.md ./

# Install dependencies using uv with CPU-only PyTorch to reduce image size
# This avoids downloading ~3GB of CUDA libraries
# Use cache mount to speed up rebuilds
ENV UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable

# Remove unnecessary files from venv to reduce image size significantly
RUN find /app/.venv -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true && \
    find /app/.venv -type f -name "*.pyc" -delete 2>/dev/null || true && \
    find /app/.venv -type f -name "*.pyo" -delete 2>/dev/null || true && \
    find /app/.venv -type d -name "tests" -exec rm -rf {} + 2>/dev/null || true && \
    find /app/.venv -type d -name "test" -exec rm -rf {} + 2>/dev/null || true && \
    find /app/.venv -type d -name "docs" -exec rm -rf {} + 2>/dev/null || true && \
    find /app/.venv -type d -name "examples" -exec rm -rf {} + 2>/dev/null || true && \
    find /app/.venv -type f -name "*.md" -delete 2>/dev/null || true && \
    find /app/.venv -type f -name "*.rst" -delete 2>/dev/null || true && \
    find /app/.venv -type f -name "*.txt" ! -name "RECORD" -delete 2>/dev/null || true && \
    find /app/.venv -type f -name "LICENSE*" -delete 2>/dev/null || true && \
    find /app/.venv -type f -name "NOTICE*" -delete 2>/dev/null || true && \
    find /app/.venv -type f -name "AUTHORS*" -delete 2>/dev/null || true && \
    find /app/.venv -type f -name "CHANGELOG*" -delete 2>/dev/null || true && \
    find /app/.venv -type f -name "*.c" -delete 2>/dev/null || true && \
    find /app/.venv -type f -name "*.h" -delete 2>/dev/null || true && \
    find /app/.venv -type f -name "*.pxd" -delete 2>/dev/null || true && \
    rm -rf /app/.venv/share/man /app/.venv/share/doc 2>/dev/null || true

# Copy source code
COPY src/ ./src/
COPY config/ ./config/

# ============================================
# Stage 2: Production Runtime
# ============================================
FROM python:${PYTHON_VERSION}-slim AS production

# Add labels for container metadata
LABEL org.opencontainers.image.source="https://github.com/ms-kumar/rag-journey-poc-to-prod"
LABEL org.opencontainers.image.description="RAG API Production Image"

# Install runtime dependencies in single layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /bin/bash appuser

# Set working directory
WORKDIR /app

# Copy virtual environment and application code with ownership set
# Using --chown avoids extra chown layer
COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv
COPY --from=builder --chown=appuser:appuser /app/src /app/src
COPY --from=builder --chown=appuser:appuser /app/config /app/config

# Switch to non-root user
USER appuser

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Default command
CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ============================================
# Stage 3: Development (optional, for local dev)
# ============================================
FROM production AS development

USER root

# Install dev dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy uv from uv stage
COPY --from=uv /uv /usr/local/bin/uv

# Install dev dependencies (README.md needed for pyproject.toml)
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-editable

USER appuser

# Override command for development with reload
CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
