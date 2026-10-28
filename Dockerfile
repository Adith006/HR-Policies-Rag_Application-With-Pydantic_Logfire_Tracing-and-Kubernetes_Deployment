
FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Copy dependency metadata first for Docker layer caching.
COPY pyproject.toml uv.lock ./

# Install dependencies exactly as locked, without installing the project itself.
RUN uv sync --frozen --no-dev --no-install-project

# Copy application source.
COPY app/ ./app/

EXPOSE 8000

CMD ["uv", "run", "--frozen", "--no-sync", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
