from __future__ import annotations

from collections.abc import Iterator
from functools import lru_cache

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import get_settings


class Base(DeclarativeBase):
    pass


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    settings = get_settings()
    return create_engine(
        settings.database_url,
        pool_pre_ping=True,
    )


@lru_cache(maxsize=1)
def get_session_factory() -> sessionmaker[Session]:
    return sessionmaker(
        bind=get_engine(),
        autoflush=False,
        expire_on_commit=False,
    )


def get_db_session() -> Iterator[Session]:
    session = get_session_factory()()
    try:
        yield session
    finally:
        session.close()


def new_session() -> Session:
    return get_session_factory()()


def init_db() -> None:
    import app.models  # noqa: F401

    engine = get_engine()
    Base.metadata.create_all(bind=engine)
    _ensure_jobs_schema(engine)


def _ensure_jobs_schema(engine: Engine) -> None:
    inspector = inspect(engine)
    if "jobs" not in inspector.get_table_names():
        return

    existing_columns = {column["name"] for column in inspector.get_columns("jobs")}
    statements: list[str] = []

    if "output_csv_key" not in existing_columns:
        statements.append("ALTER TABLE jobs ADD COLUMN output_csv_key VARCHAR(1024)")
    if "preview_json_key" not in existing_columns:
        statements.append("ALTER TABLE jobs ADD COLUMN preview_json_key VARCHAR(1024)")
    if "crop_keys_json" not in existing_columns:
        statements.append("ALTER TABLE jobs ADD COLUMN crop_keys_json JSON")
    if "stats_json" not in existing_columns:
        statements.append("ALTER TABLE jobs ADD COLUMN stats_json JSON")
    if "pipeline_name" not in existing_columns:
        statements.append(
            "ALTER TABLE jobs ADD COLUMN pipeline_name VARCHAR(128) NOT NULL DEFAULT 'price_tag_cpu_v1'"
        )
    if "pipeline_version" not in existing_columns:
        statements.append(
            "ALTER TABLE jobs ADD COLUMN pipeline_version VARCHAR(64) NOT NULL DEFAULT '0.1.0'"
        )

    if not statements:
        return

    with engine.begin() as connection:
        for statement in statements:
            connection.execute(text(statement))
