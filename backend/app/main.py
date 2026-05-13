from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.services.status import build_core_status, build_payload, build_system_status, is_healthy

settings = get_settings()

app = FastAPI(
    title="Price Tag Vision Backend",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

if settings.cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": "backend",
        "status_url": "/api/system/status",
        "health_url": "/health",
    }


@app.get("/health")
def health() -> JSONResponse:
    services = build_core_status(settings)
    payload = build_payload("backend", services)
    return JSONResponse(payload, status_code=200 if is_healthy(services) else 503)


@app.get("/api/system/status")
def system_status() -> JSONResponse:
    services = build_system_status(settings)
    payload = build_payload("backend", services)
    return JSONResponse(payload, status_code=200 if is_healthy(services) else 503)
