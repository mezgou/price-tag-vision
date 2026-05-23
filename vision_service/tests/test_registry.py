from __future__ import annotations

from app.pipelines.registry import get_pipeline_registry


def test_registry_resolves_known_pipeline_and_safe_fallback() -> None:
    registry = get_pipeline_registry()

    assert registry.get("mock") is not None
    assert registry.get("price_tag_cpu_v1") is not None
    assert registry.get("price_tag_v2") is not None
    assert registry.get("price_tag_v5") is not None
    assert registry.get("lenta_cpu_v1") is None
    assert registry.resolve("price_tag_cpu_v1", default_name="mock").name == "price_tag_cpu_v1"
    assert registry.resolve("price_tag_v2", default_name="mock").name == "price_tag_v2"
    assert registry.resolve("price_tag_v5", default_name="mock").name == "price_tag_v5"
    assert registry.resolve("unknown-pipeline", default_name="mock").name == "mock"
