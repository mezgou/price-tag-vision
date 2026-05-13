FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

ENV PYTHONPATH=/app/backend:/app
ENV UV_LINK_MODE=copy
ENV UV_PROJECT_ENVIRONMENT=/opt/venv

COPY backend/pyproject.toml backend/uv.lock* /app/backend/
COPY pyproject.toml uv.lock /app/
COPY vision_service/pyproject.toml /app/vision_service/
RUN uv sync --frozen --project /app/backend --no-install-project

COPY backend /app/backend
COPY shared /app/shared

CMD ["uv", "run", "--project", "/app/backend", "--no-sync", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
