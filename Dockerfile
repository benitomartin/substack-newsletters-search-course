# ---------- Build Stage ----------
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

WORKDIR /app

# Configure UV for optimal performance
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV UV_PYTHON_DOWNLOADS=never

# Copy dependency files and sync dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-dev

# Copy source code selectively
COPY src/api ./src/api
COPY src/config.py ./src/config.py
COPY src/infrastructure/qdrant ./src/infrastructure/qdrant
COPY src/models ./src/models
COPY src/utils ./src/utils

# Also copy README.md, pyproject.toml and uv.lock for the final sync
COPY pyproject.toml uv.lock README.md ./

# Install project dependencies into virtualenv
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev


# ---------- Runtime Stage ----------
FROM python:3.12-slim-bookworm

# Copy built application and virtualenv from builder
COPY --from=builder /app /app

# Set Python path and environment variables
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH=/app
ENV HF_HOME=/tmp/huggingface
ENV FASTEMBED_CACHE=/tmp/fastembed_cache
ENV PORT=8080

# Create cache directories
RUN mkdir -p $HF_HOME $FASTEMBED_CACHE && chmod -R 755 $HF_HOME $FASTEMBED_CACHE

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:$PORT/health || exit 1

# Expose Cloud Run port
EXPOSE $PORT

# Run FastAPI with uvicorn
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1", "--loop", "uvloop"]
