"""Metric calibration: printed checkerboard -> homography -> top-down image with a fixed px/mm.

The checkerboard is only used to calibrate the camera (as in any metrology set-up); it is not used
to find or identify gloves.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


class Rectifier:
    def __init__(self, H: np.ndarray, size: tuple[int, int], px_per_mm: float, calibrated: bool = True,
                 rms_mm: float | None = None):
        self.H = np.asarray(H, np.float64)  # camera px -> rectified px
        self.Hinv = np.linalg.inv(self.H)
        self.size = (int(size[0]), int(size[1]))
        self.px_per_mm = float(px_per_mm)
        self.calibrated = calibrated
        self.rms_mm = rms_mm
        self._valid = None
        self._valid_for = None

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_checkerboard(cls, img: np.ndarray, cols: int, rows: int, square_mm: float, px_per_mm: float) -> "Rectifier":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        ok, corners = cv2.findChessboardCornersSB(gray, (cols, rows), flags=cv2.CALIB_CB_NORMALIZE_IMAGE)
        if not ok:
            ok, corners = cv2.findChessboardCorners(gray, (cols, rows))
            if not ok:
                raise RuntimeError("checkerboard not found")
            corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1),
                                       (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.01))
        corners = corners.reshape(-1, 2).astype(np.float64)
        if np.hypot(*corners[0]) > np.hypot(*corners[-1]):  # make the first corner the top-left one
            corners = corners[::-1].copy()
        obj = np.array([[i * square_mm, j * square_mm] for j in range(rows) for i in range(cols)], np.float64)
        Hb, _ = cv2.findHomography(corners, obj)
        proj = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), Hb).reshape(-1, 2)
        rms = float(np.sqrt(np.mean(np.sum((proj - obj) ** 2, axis=1))))
        h, w = gray.shape
        img_corners = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float64).reshape(-1, 1, 2)
        mm = cv2.perspectiveTransform(img_corners, Hb).reshape(-1, 2)
        lo = mm.min(axis=0)
        hi = mm.max(axis=0)
        S = np.array([[px_per_mm, 0, -lo[0] * px_per_mm], [0, px_per_mm, -lo[1] * px_per_mm], [0, 0, 1]])
        size = tuple(int(np.ceil(v)) for v in (hi - lo) * px_per_mm)
        return cls(S @ Hb, size, px_per_mm, True, rms)

    @classmethod
    def uncalibrated(cls, shape, px_per_mm: float) -> "Rectifier":
        """Identity mapping with an assumed scale (for quick tests without a checkerboard)."""
        h, w = shape[:2]
        return cls(np.eye(3), (w, h), px_per_mm, calibrated=False)

    # ------------------------------------------------------------------ use
    def warp(self, img: np.ndarray, interp=cv2.INTER_LINEAR) -> np.ndarray:
        if not self.calibrated and img.shape[1] == self.size[0] and img.shape[0] == self.size[1]:
            return img
        return cv2.warpPerspective(img, self.H, self.size, flags=interp, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    def valid_mask(self, cam_shape) -> np.ndarray:
        """Pixels of the rectified image that the camera actually sees (eroded a little)."""
        if self._valid is None or self._valid_for != cam_shape[:2]:
            ones = np.full(cam_shape[:2], 255, np.uint8)
            v = cv2.warpPerspective(ones, self.H, self.size, flags=cv2.INTER_NEAREST, borderValue=0)
            self._valid = cv2.erode(v, np.ones((7, 7), np.uint8))
            self._valid_for = cam_shape[:2]
        return self._valid

    def img_to_rect(self, pts):
        pts = np.asarray(pts, np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.H).reshape(-1, 2)

    def rect_to_img(self, pts):
        pts = np.asarray(pts, np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.Hinv).reshape(-1, 2)

    # ------------------------------------------------------------------ persistence
    def save(self, path: str | Path):
        Path(path).write_text(json.dumps(dict(H=self.H.tolist(), size=self.size, px_per_mm=self.px_per_mm,
                                              calibrated=self.calibrated, rms_mm=self.rms_mm), indent=1))

    @classmethod
    def load(cls, path: str | Path) -> "Rectifier":
        d = json.loads(Path(path).read_text())
        return cls(np.array(d["H"]), tuple(d["size"]), d["px_per_mm"], d.get("calibrated", True), d.get("rms_mm"))
