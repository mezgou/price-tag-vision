from __future__ import annotations

from collections.abc import Sequence

from app.schemas.detections import CropCandidate


def build_crop_quality_summary(
    crops: Sequence[CropCandidate],
) -> dict[str, float | int]:
    if not crops:
        return {
            "count": 0,
            "score_min": 0.0,
            "score_max": 0.0,
            "score_mean": 0.0,
            "score_median": 0.0,
            "sharpness_mean": 0.0,
            "glare_ratio_mean": 0.0,
        }

    scores = sorted(crop.quality.score for crop in crops)
    sharpness_values = [crop.quality.sharpness for crop in crops]
    glare_values = [crop.quality.glare_ratio for crop in crops]
    count = len(scores)
    mid = count // 2

    if count % 2 == 0:
        score_median = (scores[mid - 1] + scores[mid]) / 2.0
    else:
        score_median = scores[mid]

    return {
        "count": count,
        "score_min": round(min(scores), 6),
        "score_max": round(max(scores), 6),
        "score_mean": round(sum(scores) / count, 6),
        "score_median": round(score_median, 6),
        "sharpness_mean": round(sum(sharpness_values) / count, 6),
        "glare_ratio_mean": round(sum(glare_values) / count, 6),
    }
