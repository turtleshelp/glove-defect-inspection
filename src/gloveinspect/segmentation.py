"""Lighting-robust, colour-agnostic glove segmentation.

1. The empty inspection plate is learned once (median of a few frames).
2. Every new frame is divided by a smooth per-channel gain field (a + b*x + c*y), fitted robustly on
   plate pixels. This cancels global brightness, colour temperature and linear light gradients.
3. The compensated frame is compared with the plate in CIE-LAB using a shadow-tolerant distance
   D = |chroma difference| + w * |lightness difference|; Otsu picks the threshold for each frame.
Nothing here depends on the glove's own colour.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


def to_lab(bgr_float01: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr_float01.astype(np.float32), cv2.COLOR_BGR2LAB)


@dataclass
class Segmentation:
    comp: np.ndarray        # gain-compensated frame, float32 BGR 0..1 (rectified)
    lab: np.ndarray         # LAB of comp
    dist: np.ndarray        # shadow-tolerant distance to plate
    de: np.ndarray          # full Delta-E (CIE76) to plate
    mask: np.ndarray        # uint8 0/255 foreground after cleanup
    threshold: float
    contours: list          # external contours of glove-sized blobs
    gain: np.ndarray        # fitted gain coefficients (3 x 3)
    plate_L: np.ndarray | None = None  # lightness of the learned plate


class BackgroundModel:
    def __init__(self, cfg: dict, px_per_mm: float):
        self.cfg = cfg
        self.ppm = px_per_mm
        self.bg = None
        self.bg_lab = None
        self._grid = None

    @property
    def ready(self) -> bool:
        return self.bg is not None

    def learn(self, frames_rect: list[np.ndarray]):
        stack = np.stack([f.astype(np.float32) / 255.0 for f in frames_rect])
        self.bg = np.median(stack, axis=0).astype(np.float32)
        self.bg_lab = to_lab(self.bg)

    def _fit_gain(self, f: np.ndarray, valid: np.ndarray) -> np.ndarray:
        h, w = f.shape[:2]
        step = 8
        ys, xs = np.mgrid[step // 2:h:step, step // 2:w:step]
        keep = valid[ys, xs] > 0
        ys, xs = ys[keep], xs[keep]
        X = np.stack([np.ones_like(xs, np.float64), xs / w - 0.5, ys / h - 0.5], axis=1)
        coefs = np.zeros((3, 3))
        fs = f[ys, xs].astype(np.float64)
        bs = self.bg[ys, xs].astype(np.float64)
        r_all = (fs + 0.01) / (bs + 0.01)
        inl = np.ones(len(xs), bool)
        for c in range(3):
            med = np.median(r_all[:, c])
            inl &= np.abs(r_all[:, c] - med) < 0.25 * med
        for _ in range(3):
            res_ok = np.ones(len(xs), bool)
            for c in range(3):
                r = r_all[:, c]
                sol, *_ = np.linalg.lstsq(X[inl], r[inl], rcond=None)
                coefs[c] = sol
                res = r - X @ sol
                mad = np.median(np.abs(res[inl])) * 1.4826 + 1e-3
                res_ok &= np.abs(res) < 3.0 * mad + 0.02
            inl = res_ok
        return coefs

    def segment(self, frame_rect_u8: np.ndarray, valid: np.ndarray) -> Segmentation:
        cfg = self.cfg
        f = frame_rect_u8.astype(np.float32) / 255.0
        h, w = f.shape[:2]
        coefs = self._fit_gain(f, valid)
        xs = (np.arange(w, dtype=np.float32) / w - 0.5)[None, :]
        ys = (np.arange(h, dtype=np.float32) / h - 0.5)[:, None]
        comp = np.empty_like(f)
        for c in range(3):
            field = coefs[c, 0] + coefs[c, 1] * xs + coefs[c, 2] * ys
            comp[:, :, c] = f[:, :, c] / np.maximum(field, 0.05)
        comp = np.clip(comp, 0, 1.5)
        lab = to_lab(np.clip(comp, 0, 1))
        d = lab - self.bg_lab
        chroma = np.sqrt(d[:, :, 1] ** 2 + d[:, :, 2] ** 2)
        dist = chroma + cfg["l_weight"] * np.abs(d[:, :, 0])
        de = np.sqrt(chroma ** 2 + d[:, :, 0] ** 2)
        dist = cv2.GaussianBlur(dist, (0, 0), 1.0)
        dist[valid == 0] = 0
        d8 = np.clip(dist * 4, 0, 255).astype(np.uint8)  # 0.25 units per level, saturates at 64
        thr8, _ = cv2.threshold(d8[valid > 0].reshape(1, -1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thr = max(float(thr8) / 4.0, cfg["min_contrast"])
        mask = ((dist > thr) & (valid > 0)).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        min_area = cfg["min_glove_area_mm2"] * self.ppm ** 2
        cnts = [c for c in cnts if cv2.contourArea(c) >= min_area]
        return Segmentation(comp, lab, dist, de, mask, thr, cnts, coefs, self.bg_lab[:, :, 0])

    def update(self, seg: Segmentation, valid: np.ndarray):
        """Slowly adapt the plate model where no glove is present (camera drift, dust)."""
        rate = self.cfg.get("bg_update_rate", 0.0)
        self._n_upd = getattr(self, "_n_upd", 0) + 1
        if rate <= 0 or self._n_upd % 10:  # every 10th frame is enough for slow drift
            return
        free = (cv2.dilate(seg.mask, np.ones((31, 31), np.uint8)) == 0) & (valid > 0)
        if free.mean() < 0.3:
            return
        # store the update in the plate's own lighting (undo nothing: comp is already in plate lighting)
        upd = np.clip(seg.comp, 0, 1)
        self.bg[free] = (1 - rate) * self.bg[free] + rate * upd[free]
        self.bg_lab = to_lab(self.bg)
