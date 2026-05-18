"""Camera geometry: lens-distortion correction + exact point round-trips.

The shelf camera has moderate barrel distortion. We process frames in an
**undistorted + 90deg-CCW upright** space (best for the detector, QR and OCR),
but the official CSV / hidden metric expects bounding boxes in the **raw
distorted 3840x2160** coordinate space. This module owns that whole geometry
chain and its exact inverse so detections can be mapped back to raw pixels.

Forward  (raw distorted) -> undistort+ROI -> rotate_90_ccw -> processed
Inverse  processed -> un-rotate -> add ROI -> re-distort -> raw distorted

Distortion params come from the manufacturer note (``example_undistort.py``):
intrinsics from sensor geometry, Brown-Conrady coeffs [k1,k2,p1,p2,k3].
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

# Defaults from example_undistort.py (manufacturer-provided).
DEFAULT_IMAGE_SIZE = (3840, 2160)
DEFAULT_FOCAL_MM = 2.8
DEFAULT_DIAGONAL_MM = 16.0 / 2.8  # vidicon standard conversion ~5.714 mm
DEFAULT_DIST_COEFFS = (-0.276, 0.06, 0.0084, -0.0016, -0.0044)  # k1,k2,p1,p2,k3

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class CameraGeometry:
    """Result sizes for the processed (undistorted + rotated) space."""

    roi: tuple[int, int, int, int]  # x, y, w, h of valid undistorted region
    undistorted_size: tuple[int, int]  # (w, h) after ROI crop, before rotation
    processed_size: tuple[int, int]  # (w, h) after rotate_90_ccw


class CameraModel:
    """Distortion model + the raw<->processed geometry chain.

    All point methods accept/return float (x, y) arrays of shape (N, 2).
    """

    def __init__(
        self,
        image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
        focal_mm: float = DEFAULT_FOCAL_MM,
        diagonal_mm: float = DEFAULT_DIAGONAL_MM,
        dist_coeffs: tuple[float, ...] = DEFAULT_DIST_COEFFS,
        *,
        rotate_ccw: bool = True,
    ) -> None:
        self.width, self.height = image_size
        self.rotate_ccw = rotate_ccw
        self.dist = np.asarray(dist_coeffs, dtype=np.float64).reshape(-1)

        aspect = self.width / self.height
        h_mm = diagonal_mm / math.sqrt(aspect**2 + 1.0)
        w_mm = aspect * h_mm
        fx = focal_mm * self.width / w_mm
        fy = focal_mm * self.height / h_mm
        self.K = np.array(
            [[fx, 0.0, self.width / 2.0],
             [0.0, fy, self.height / 2.0],
             [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        self.new_K, roi = cv2.getOptimalNewCameraMatrix(
            self.K, self.dist, (self.width, self.height), 0,
            (self.width, self.height),
        )
        self.new_K = np.asarray(self.new_K, dtype=np.float64)
        rx, ry, rw, rh = (int(v) for v in roi)
        # Guard against the occasional degenerate 0-size ROI.
        if rw <= 0 or rh <= 0:
            rx, ry, rw, rh = 0, 0, self.width, self.height
        self.roi = (rx, ry, rw, rh)
        self._inv_new_K = np.linalg.inv(self.new_K)

        self._map1, self._map2 = cv2.initUndistortRectifyMap(
            self.K, self.dist, None, self.new_K,
            (self.width, self.height), cv2.CV_16SC2,
        )
        und_w, und_h = rw, rh
        proc_w, proc_h = (und_h, und_w) if rotate_ccw else (und_w, und_h)
        self.geometry = CameraGeometry(
            roi=self.roi,
            undistorted_size=(und_w, und_h),
            processed_size=(proc_w, proc_h),
        )

    # ----- image transforms -------------------------------------------------
    def undistort_image(self, frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
        und = cv2.remap(frame, self._map1, self._map2, cv2.INTER_LINEAR)
        rx, ry, rw, rh = self.roi
        return und[ry:ry + rh, rx:rx + rw]

    def raw_frame_to_processed(self, frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
        und = self.undistort_image(frame)
        if self.rotate_ccw:
            return cv2.rotate(und, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return und

    # ----- point transforms -------------------------------------------------
    def distorted_to_undistorted(self, pts: FloatArray) -> FloatArray:
        """Raw distorted pixels -> undistorted (ROI-cropped) pixels."""
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
        criteria = (
            cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 100, 1e-7
        )
        und = cv2.undistortPointsIter(
            pts.astype(np.float32), self.K.astype(np.float32),
            self.dist.astype(np.float32), None,
            self.new_K.astype(np.float32), criteria,
        ).reshape(-1, 2).astype(np.float64)
        rx, ry, _, _ = self.roi
        und[:, 0] -= rx
        und[:, 1] -= ry
        return und

    def undistorted_to_distorted(self, pts: FloatArray) -> FloatArray:
        """Undistorted (ROI-cropped) pixels -> raw distorted pixels.

        Analytic inverse of ``distorted_to_undistorted`` (Brown-Conrady
        forward projection), accurate to well under a pixel.
        """
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        rx, ry, _, _ = self.roi
        full = np.empty((pts.shape[0], 3), dtype=np.float64)
        full[:, 0] = pts[:, 0] + rx
        full[:, 1] = pts[:, 1] + ry
        full[:, 2] = 1.0
        norm = (self._inv_new_K @ full.T).T  # normalized camera rays (z=1)
        proj, _ = cv2.projectPoints(
            norm.reshape(-1, 1, 3).astype(np.float64),
            np.zeros(3), np.zeros(3),
            self.K, self.dist,
        )
        return proj.reshape(-1, 2).astype(np.float64)

    # ----- rotation helpers -------------------------------------------------
    def _rot_ccw(self, pts: FloatArray) -> FloatArray:
        und_w, _ = self.geometry.undistorted_size
        out = np.empty_like(pts)
        out[:, 0] = pts[:, 1]
        out[:, 1] = und_w - pts[:, 0]
        return out

    def _unrot_ccw(self, pts: FloatArray) -> FloatArray:
        und_w, _ = self.geometry.undistorted_size
        out = np.empty_like(pts)
        out[:, 0] = und_w - pts[:, 1]
        out[:, 1] = pts[:, 0]
        return out

    # ----- bbox chain (axis-aligned xyxy) ----------------------------------
    @staticmethod
    def _corners(box: tuple[float, float, float, float]) -> FloatArray:
        x1, y1, x2, y2 = box
        return np.array(
            [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float64
        )

    @staticmethod
    def _aabb(pts: FloatArray) -> tuple[float, float, float, float]:
        xs, ys = pts[:, 0], pts[:, 1]
        return (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))

    def raw_box_to_processed(
        self, box: tuple[float, float, float, float]
    ) -> tuple[float, float, float, float]:
        """Raw distorted xyxy -> processed (undistorted + ccw) xyxy.

        Used by the dataset builder to transform ground-truth labels.
        """
        pts = self.distorted_to_undistorted(self._corners(box))
        if self.rotate_ccw:
            pts = self._rot_ccw(pts)
        return self._aabb(pts)

    def processed_box_to_raw(
        self, box: tuple[float, float, float, float]
    ) -> tuple[float, float, float, float]:
        """Processed (undistorted + ccw) xyxy -> raw distorted xyxy.

        Used at CSV time so bounding boxes match the hidden-metric GT space.
        Clamped to the raw frame bounds.
        """
        pts = self._corners(box)
        if self.rotate_ccw:
            pts = self._unrot_ccw(pts)
        raw = self.undistorted_to_distorted(pts)
        raw[:, 0] = np.clip(raw[:, 0], 0.0, self.width - 1.0)
        raw[:, 1] = np.clip(raw[:, 1], 0.0, self.height - 1.0)
        return self._aabb(raw)
