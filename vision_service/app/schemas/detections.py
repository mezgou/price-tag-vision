from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class BoundingBox(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x_min: int
    y_min: int
    x_max: int
    y_max: int

    @model_validator(mode="after")
    def validate_coordinates(self) -> "BoundingBox":
        if self.x_min < 0 or self.y_min < 0:
            raise ValueError("Bounding box coordinates must be non-negative.")
        if self.x_max <= self.x_min:
            raise ValueError("x_max must be greater than x_min.")
        if self.y_max <= self.y_min:
            raise ValueError("y_max must be greater than y_min.")
        return self

    @property
    def width(self) -> int:
        return self.x_max - self.x_min

    @property
    def height(self) -> int:
        return self.y_max - self.y_min

    @property
    def area(self) -> int:
        return self.width * self.height


class DetectionCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    detection_id: str
    frame_index: int
    timestamp_ms: int | None
    label: str
    bbox: BoundingBox
    confidence: float = Field(ge=0.0, le=1.0)
    source: str
    attributes: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_identity(self) -> "DetectionCandidate":
        if not self.detection_id.strip():
            raise ValueError("detection_id must not be empty.")
        if not self.label.strip():
            raise ValueError("label must not be empty.")
        if not self.source.strip():
            raise ValueError("source must not be empty.")
        if self.frame_index < 0:
            raise ValueError("frame_index must be non-negative.")
        if self.timestamp_ms is not None and self.timestamp_ms < 0:
            raise ValueError("timestamp_ms must be non-negative when provided.")
        return self


class CropQuality(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sharpness: float = Field(ge=0.0)
    brightness: float = Field(ge=0.0)
    contrast: float = Field(ge=0.0)
    glare_ratio: float = Field(ge=0.0, le=1.0)
    area_ratio: float = Field(ge=0.0, le=1.0)
    score: float = Field(ge=0.0, le=1.0)


class CropCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    crop_id: str
    detection_id: str
    frame_index: int
    timestamp_ms: int | None
    bbox: BoundingBox
    padded_bbox: BoundingBox
    crop_key: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    quality: CropQuality
    source: str
    attributes: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_identity(self) -> "CropCandidate":
        if not self.crop_id.strip():
            raise ValueError("crop_id must not be empty.")
        if not self.detection_id.strip():
            raise ValueError("detection_id must not be empty.")
        if not self.source.strip():
            raise ValueError("source must not be empty.")
        if self.frame_index < 0:
            raise ValueError("frame_index must be non-negative.")
        if self.timestamp_ms is not None and self.timestamp_ms < 0:
            raise ValueError("timestamp_ms must be non-negative when provided.")
        return self


class DecodeAttempt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    attempt_id: str
    crop_id: str
    decoder: str
    variant: str
    success: bool
    error: str | None = None
    duration_ms: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_identity(self) -> "DecodeAttempt":
        if not self.attempt_id.strip():
            raise ValueError("attempt_id must not be empty.")
        if not self.crop_id.strip():
            raise ValueError("crop_id must not be empty.")
        if not self.decoder.strip():
            raise ValueError("decoder must not be empty.")
        if not self.variant.strip():
            raise ValueError("variant must not be empty.")
        return self


class DecodedSymbol(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol_id: str
    crop_id: str
    detection_id: str
    frame_index: int
    timestamp_ms: int | None
    symbol_type: str
    decoder: str
    variant: str
    payload: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: BoundingBox | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_identity(self) -> "DecodedSymbol":
        if not self.symbol_id.strip():
            raise ValueError("symbol_id must not be empty.")
        if not self.crop_id.strip():
            raise ValueError("crop_id must not be empty.")
        if not self.detection_id.strip():
            raise ValueError("detection_id must not be empty.")
        if not self.decoder.strip():
            raise ValueError("decoder must not be empty.")
        if not self.variant.strip():
            raise ValueError("variant must not be empty.")
        if not self.payload.strip():
            raise ValueError("payload must not be empty.")
        if self.frame_index < 0:
            raise ValueError("frame_index must be non-negative.")
        if self.timestamp_ms is not None and self.timestamp_ms < 0:
            raise ValueError("timestamp_ms must be non-negative when provided.")
        if self.symbol_type not in {"qr", "barcode", "unknown"}:
            raise ValueError("symbol_type must be one of: qr, barcode, unknown.")
        return self
