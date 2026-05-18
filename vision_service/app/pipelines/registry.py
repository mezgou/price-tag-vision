from __future__ import annotations

from functools import lru_cache

from app.pipelines.base import BasePipeline
from app.pipelines.mock import MockPipeline
from app.pipelines.price_tag_cpu_v1 import PriceTagCpuV1Pipeline
from app.pipelines.price_tag_v2 import PriceTagV2Pipeline
from app.pipelines.price_tag_v3 import PriceTagV3Pipeline


class PipelineRegistry:
    def __init__(self) -> None:
        self._pipelines: dict[str, BasePipeline] = {}

    def register(self, pipeline: BasePipeline) -> None:
        self._pipelines[pipeline.name] = pipeline

    def get(self, name: str) -> BasePipeline | None:
        return self._pipelines.get(name)

    def resolve(self, name: str | None, *, default_name: str) -> BasePipeline:
        requested_name = name or default_name

        if requested_name in self._pipelines:
            return self._pipelines[requested_name]
        if default_name in self._pipelines:
            return self._pipelines[default_name]
        if "mock" in self._pipelines:
            return self._pipelines["mock"]

        available = ", ".join(sorted(self._pipelines))
        raise KeyError(f"Pipeline '{requested_name}' is not registered. Available: {available}")

    def names(self) -> list[str]:
        return sorted(self._pipelines)


@lru_cache(maxsize=1)
def get_pipeline_registry() -> PipelineRegistry:
    registry = PipelineRegistry()
    registry.register(MockPipeline())
    registry.register(PriceTagCpuV1Pipeline())
    registry.register(PriceTagV2Pipeline())
    registry.register(PriceTagV3Pipeline())
    return registry
