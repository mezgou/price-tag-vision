from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

from app.schemas.detections import DetectionCandidate

FrameArray = NDArray[np.uint8]
BOX_COLOR = (0, 210, 255)
TEXT_COLOR = (20, 20, 20)
TEXT_BG_COLOR = (255, 255, 255)


def draw_detection_candidates(
    frame: FrameArray,
    detections: Sequence[DetectionCandidate],
) -> FrameArray:
    annotated = frame.copy()

    for index, detection in enumerate(detections, start=1):
        bbox = detection.bbox
        cv2.rectangle(
            annotated,
            (bbox.x_min, bbox.y_min),
            (bbox.x_max, bbox.y_max),
            BOX_COLOR,
            2,
        )

        label = f"{index}:{detection.confidence:.2f}"
        (label_width, label_height), _ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            1,
        )
        text_origin_x = bbox.x_min
        text_origin_y = max(bbox.y_min - 6, label_height + 6)
        text_box_top_left = (text_origin_x, text_origin_y - label_height - 4)
        text_box_bottom_right = (text_origin_x + label_width + 6, text_origin_y + 2)
        cv2.rectangle(
            annotated,
            text_box_top_left,
            text_box_bottom_right,
            TEXT_BG_COLOR,
            thickness=-1,
        )
        cv2.putText(
            annotated,
            label,
            (text_origin_x + 3, text_origin_y - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            TEXT_COLOR,
            1,
            cv2.LINE_AA,
        )

    return annotated
