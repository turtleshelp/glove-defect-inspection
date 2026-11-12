"""Scene rendering: inspection plate, gloves, calibration checkerboard, camera and lighting."""
from __future__ import annotations

import math

import cv2
import numpy as np

from .model import K, delta_e

WORLD_PPM = 2.0  # world raster, px per mm
WORLD_W, WORLD_H = 1700, 1000  # 850 x 500 mm plate
CAM_W, CAM_H = 1600, 900

BOARD = dict(cols=9, rows=6, square_mm=20.0, margin_mm=10.0)  # inner corners

GLOVE_COLOURS_BGR = {
    "blue nitrile": (0.75, 0.40, 0.10),
    "purple nitrile": (0.60, 0.25, 0.45),
    "light blue": (0.85, 0.65, 0.35),
    "green": (0.30, 0.55, 0.20),
    "orange": (0.10, 0.45, 0.90),
    "pink": (0.65, 0.55, 0.90),
    "natural latex": (0.70, 0.86, 0.92),
}

LIGHTING = {
    "nominal": dict(gain=1.0, col=(1.0, 1.0, 1.0), gx=0.0, gy=0.0, shadow=False, noise=2.0, blur=0.0, jpeg=92),
    "dim": dict(gain=0.6, col=(1.0, 1.0, 1.0), gx=0.0, gy=0.0, shadow=False, noise=2.5, blur=0.0, jpeg=92),
    "bright": dict(gain=1.3, col=(1.0, 1.0, 1.0), gx=0.0, gy=0.0, shadow=False, noise=2.0, blur=0.0, jpeg=92),
    "warm_gradient": dict(gain=1.0, col=(0.82, 0.97, 1.12), gx=0.35, gy=-0.2, shadow=False, noise=2.5, blur=0.0, jpeg=90),
    "shadow_noise": dict(gain=0.9, col=(1.0, 1.0, 1.0), gx=0.0, gy=0.0, shadow=True, noise=6.0, blur=1.0, jpeg=75),
}


def segmentation_contrast(c1, c2) -> float:
    """Shadow-tolerant colour distance used by the inspector: chroma distance + 0.5 * |dL|."""
    a = cv2.cvtColor(np.array([[c1]], np.float32), cv2.COLOR_BGR2LAB)[0, 0]
    b = cv2.cvtColor(np.array([[c2]], np.float32), cv2.COLOR_BGR2LAB)[0, 0]
    return float(np.hypot(a[1] - b[1], a[2] - b[2]) + 0.5 * abs(a[0] - b[0]))


class Scene:
    def __init__(self, rng: np.random.Generator, plate_bgr=(0.46, 0.47, 0.48)):
        self.rng = rng
        inset = 50
        src = np.float32([[inset, inset], [WORLD_W - inset, inset], [WORLD_W - inset, WORLD_H - inset], [inset, WORLD_H - inset]])
        src = src + rng.uniform(-20, 20, src.shape).astype(np.float32)
        dst = np.float32([[0, 0], [CAM_W, 0], [CAM_W, CAM_H], [0, CAM_H]])
        self.H = cv2.getPerspectiveTransform(src, dst)
        self.plate_bgr = plate_bgr
        base = np.array(plate_bgr, np.float32)
        tex = cv2.GaussianBlur(rng.normal(0, 0.02, (WORLD_H, WORLD_W)).astype(np.float32), (0, 0), 0.8)
        low = cv2.resize(rng.random((5, 8)).astype(np.float32), (WORLD_W, WORLD_H), interpolation=cv2.INTER_CUBIC)
        self.plate = base[None, None, :] * (1 + tex + 0.03 * (low - 0.5))[:, :, None]
        yy, xx = np.mgrid[0:CAM_H, 0:CAM_W].astype(np.float32)
        self.X, self.Y = xx / CAM_W, yy / CAM_H
        r2 = ((xx - CAM_W / 2) / (CAM_W / 2)) ** 2 + ((yy - CAM_H / 2) / (CAM_H / 2)) ** 2
        self.vignette = (1 - 0.06 * r2).astype(np.float32)

    # ------------------------------------------------------------------ geometry
    def to_camera(self, world_img):
        return cv2.warpPerspective(world_img, self.H, (CAM_W, CAM_H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    def world_to_cam(self, pts):
        pts = np.asarray(pts, np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.H).reshape(-1, 2)

    def sample_glove_colour(self, min_contrast=25.0):
        names = list(GLOVE_COLOURS_BGR)
        for _ in range(100):
            name = names[int(self.rng.integers(len(names)))]
            c = np.clip(np.array(GLOVE_COLOURS_BGR[name]) * self.rng.uniform(0.92, 1.08, 3), 0.02, 0.98)
            if segmentation_contrast(tuple(c), self.plate_bgr) >= min_contrast:
                return name, tuple(float(v) for v in c)
        raise RuntimeError("no glove colour contrasts with the plate")

    # ------------------------------------------------------------------ composition
    @staticmethod
    def glove_affine(glove: dict, center_world, angle_deg):
        """2x3 affine local px -> world px. Positive angle rotates the glove clockwise on screen."""
        cv = glove["canvas"]
        alpha = glove["alpha"]
        ys, xs = np.nonzero(alpha > 0.5)
        c_local = np.array([(xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2], np.float64)
        s = WORLD_PPM / K
        t = math.radians(angle_deg)
        R = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]]) * s
        tr = np.asarray(center_world, np.float64) - R @ c_local
        return np.hstack([R, tr[:, None]]).astype(np.float64), cv

    def composite(self, world, glove: dict, M):
        rgb = cv2.GaussianBlur(glove["rgb"], (0, 0), 0.8)
        a = cv2.GaussianBlur(glove["alpha"], (0, 0), 0.8)
        size = (world.shape[1], world.shape[0])
        rgb_w = cv2.warpAffine(rgb, M, size, flags=cv2.INTER_LINEAR, borderValue=0)
        a_w = cv2.warpAffine(a, M, size, flags=cv2.INTER_LINEAR, borderValue=0)
        sh = np.float32([[1, 0, 3 * WORLD_PPM], [0, 1, 3 * WORLD_PPM]])
        shadow = cv2.GaussianBlur(cv2.warpAffine(a_w, sh, size), (0, 0), 4 * WORLD_PPM)
        world = world * (1 - 0.12 * shadow * (1 - a_w))[:, :, None]
        world = world * (1 - a_w[:, :, None]) + rgb_w * a_w[:, :, None]
        return world, a_w

    def local_to_cam(self, glove: dict, M, pts_mm):
        cv = glove["canvas"]
        out = []
        for x, y in pts_mm:
            u, v = cv.px(x, y)
            wx, wy = M @ np.array([u, v, 1.0])
            out.append((wx, wy))
        return self.world_to_cam(out) if out else np.zeros((0, 2), np.float32)

    def calibration_world(self):
        b = BOARD
        ss = 8.0
        wmm = (b["cols"] + 1) * b["square_mm"] + 2 * b["margin_mm"]
        hmm = (b["rows"] + 1) * b["square_mm"] + 2 * b["margin_mm"]
        img = np.ones((int(hmm * ss), int(wmm * ss)), np.float32) * 0.92
        for j in range(b["rows"] + 1):
            for i in range(b["cols"] + 1):
                if (i + j) % 2 == 0:
                    x0 = (b["margin_mm"] + i * b["square_mm"]) * ss
                    y0 = (b["margin_mm"] + j * b["square_mm"]) * ss
                    img[int(y0):int(y0 + b["square_mm"] * ss), int(x0):int(x0 + b["square_mm"] * ss)] = 0.08
        ang = math.radians(self.rng.uniform(-4, 4))
        s = WORLD_PPM / ss
        R = np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]]) * s
        cw = np.array([WORLD_W / 2 + self.rng.uniform(-150, 150), WORLD_H / 2 + self.rng.uniform(-80, 80)])
        c = np.array([img.shape[1] / 2, img.shape[0] / 2])
        M = np.hstack([R, (cw - R @ c)[:, None]])
        img = cv2.GaussianBlur(img, (0, 0), ss / WORLD_PPM / 2)
        size = (WORLD_W, WORLD_H)
        bw = cv2.warpAffine(img, M, size, flags=cv2.INTER_LINEAR)
        ba = cv2.warpAffine(np.ones_like(img), M, size, flags=cv2.INTER_LINEAR)
        world = self.plate * (1 - ba[:, :, None]) + bw[:, :, None] * ba[:, :, None]
        return world

    # ------------------------------------------------------------------ camera + light
    def finish(self, cam, cond: dict, rng=None):
        rng = rng or self.rng
        f = cond["gain"] * (1 + cond["gx"] * (self.X - 0.5) + cond["gy"] * (self.Y - 0.5)) * self.vignette
        img = cam * f[:, :, None] * np.array(cond["col"], np.float32)[None, None, :]
        if cond.get("shadow"):
            m = np.zeros((CAM_H, CAM_W), np.float32)
            c = (int(rng.uniform(300, 1300)), int(rng.uniform(200, 700)))
            cv2.ellipse(m, c, (int(rng.uniform(200, 350)), int(rng.uniform(120, 220))), float(rng.uniform(0, 180)),
                        0, 360, 1.0, -1)
            m = cv2.GaussianBlur(m, (0, 0), 50)
            img = img * (1 - 0.35 * m)[:, :, None]
        img = np.clip(img * 255.0, 0, 255)
        if cond.get("blur", 0) > 0:
            img = cv2.GaussianBlur(img, (0, 0), cond["blur"])
        img = img + rng.normal(0, cond["noise"], img.shape)
        img = np.clip(img, 0, 255).astype(np.uint8)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(cond["jpeg"])])
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)

    def jitter(self, cond: dict) -> dict:
        c = dict(cond)
        c["gain"] = cond["gain"] * self.rng.uniform(0.95, 1.05)
        if cond["gx"]:
            c["gx"] = cond["gx"] * (1 if self.rng.random() < 0.5 else -1)
        return c


__all__ = ["Scene", "LIGHTING", "BOARD", "WORLD_PPM", "CAM_W", "CAM_H", "delta_e", "segmentation_contrast"]
