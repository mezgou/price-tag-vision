FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

ENV PYTHONPATH=/app/vision_service:/app
ENV UV_LINK_MODE=copy
ENV UV_PROJECT_ENVIRONMENT=/opt/venv

COPY vision_service/pyproject.toml vision_service/uv.lock* /app/vision_service/
COPY pyproject.toml uv.lock /app/
COPY backend/pyproject.toml /app/backend/
RUN uv sync --frozen --project /app/vision_service --no-install-project

COPY vision_service /app/vision_service
COPY shared /app/shared

CMD ["uv", "run", "--project", "/app/vision_service", "--no-sync", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "9001"]
