"""Prototype + verify the full geometry chain.

raw(distorted) --undistort+ROI--> undistorted --rot90ccw--> upright
and the inverse:  upright --> undistorted --> raw(distorted)

Verifies on the doc reference tag (26_12-20 @ frame 132, raw bbox
2011,1923->2231,2115 = "MOULIN DE LA FAYE -38% 2345").
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import cv2
import numpy as np

R = Path.cwd()
sys.path.insert(0, str(R / "vision_service"))
sys.path.insert(1, str(R))

IMG_W, IMG_H = 3840, 2160
FOCAL_MM = 2.8
DIAG_MM = 16.0 / 2.8
DIST = np.array([-0.276, 0.06, 0.0084, -0.0016, -0.0044], dtype=np.float32)

OUT = R / "artifacts/orient_check"


def camera_matrix() -> np.ndarray:
    aspect = IMG_W / IMG_H
    h_mm = DIAG_MM / math.sqrt(aspect**2 + 1)
    w_mm = aspect * h_mm
    fx = FOCAL_MM * IMG_W / w_mm
    fy = FOCAL_MM * IMG_H / h_mm
    return np.array([[fx, 0, IMG_W / 2], [0, fy, IMG_H / 2], [0, 0, 1]], dtype=np.float32)


def main() -> int:
    K = camera_matrix()
    newK, roi = cv2.getOptimalNewCameraMatrix(K, DIST, (IMG_W, IMG_H), 0, (IMG_W, IMG_H))
    map1, map2 = cv2.initUndistortRectifyMap(K, DIST, None, newK, (IMG_W, IMG_H), cv2.CV_32FC1)
    rx, ry, rw, rh = roi
    print("K=\n", K)
    print("roi=", roi, "newK=\n", newK)

    cap = cv2.VideoCapture("data/videos/26_12-20/26_12-20.mp4")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 132)
    ok, frame = cap.read()
    cap.release()
    assert ok

    # forward: undistort whole frame, crop ROI
    und = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
    und_c = und[ry:ry + rh, rx:rx + rw]

    # GT box corners in raw distorted space
    raw_box = np.array([[2011.9, 1923.3], [2231.6, 1923.3],
                        [2231.6, 2115.3], [2011.9, 2115.3]], dtype=np.float32)

    # distorted -> undistorted (full), then subtract ROI offset
    undp = cv2.undistortPoints(raw_box.reshape(-1, 1, 2), K, DIST, P=newK).reshape(-1, 2)
    undp_c = undp - np.array([rx, ry], dtype=np.float32)
    ux1, uy1 = undp_c.min(axis=0)
    ux2, uy2 = undp_c.max(axis=0)
    print("undistorted box (cropped):", (ux1, uy1, ux2, uy2), "und_c size", und_c.shape[1::-1])

    # rotate ccw: image (W,H)->(H,W); point (x,y)->(y, W-x) where W=und_c width
    Wc, Hc = und_c.shape[1], und_c.shape[0]
    rot = cv2.rotate(und_c, cv2.ROTATE_90_COUNTERCLOCKWISE)
    corners = [(ux1, uy1), (ux2, uy1), (ux2, uy2), (ux1, uy2)]
    rc = [(y, Wc - x) for x, y in corners]
    rxs = [p[0] for p in rc]
    rys = [p[1] for p in rc]
    rbx1, rby1, rbx2, rby2 = min(rxs), min(rys), max(rxs), max(rys)
    print("rotated box:", (rbx1, rby1, rbx2, rby2), "rot size", rot.shape[1::-1])

    crop = rot[int(rby1):int(rby2), int(rbx1):int(rbx2)]
    cv2.imwrite(str(OUT / "ref_crop_undistort_ccw.jpg"),
                cv2.resize(crop, None, fx=3, fy=3))
    full = rot.copy()
    cv2.rectangle(full, (int(rbx1), int(rby1)), (int(rbx2), int(rby2)), (0, 0, 255), 6)
    sc = 1000.0 / max(full.shape[:2])
    cv2.imwrite(str(OUT / "ref_full_undistort_ccw.jpg"),
                cv2.resize(full, None, fx=sc, fy=sc))

    # ===== INVERSE round-trip: rotated box -> undistorted -> raw distorted =====
    # inverse rotate ccw: (x',y') -> (Wc - y', x')
    inv_corners = [(rbx1, rby1), (rbx2, rby1), (rbx2, rby2), (rbx1, rby2)]
    back_und_c = [(Wc - y, x) for x, y in inv_corners]
    # add ROI offset back -> undistorted full coords
    back_und_full = [(x + rx, y + ry) for x, y in back_und_c]
    # undistorted full -> distorted via map lookup
    raw_pts = []
    for (x, y) in back_und_full:
        xi = int(np.clip(round(x), 0, IMG_W - 1))
        yi = int(np.clip(round(y), 0, IMG_H - 1))
        raw_pts.append((float(map1[yi, xi]), float(map2[yi, xi])))
    rxs = [p[0] for p in raw_pts]
    rys = [p[1] for p in raw_pts]
    got = (min(rxs), min(rys), max(rxs), max(rys))
    want = (2011.9, 1923.3, 2231.6, 2115.3)
    err = max(abs(g - w) for g, w in zip(got, want))
    print("round-trip raw box:", tuple(round(v, 1) for v in got))
    print("expected         :", want)
    print("max corner error (px):", round(err, 2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
