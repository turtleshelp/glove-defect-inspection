"""Parametric glove model and defect injection.

Local glove frame (millimetres): x to the right, y up, origin at the centre of the cuff end.
Gloves are modelled as a RIGHT hand seen from the back (thumb on the left, fingers up);
left hands are produced by mirroring x. Everything is rasterised at K px/mm.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

import cv2
import numpy as np

K = 4.0  # local raster resolution, px per mm
SHIFT = 4  # sub-pixel bits for OpenCV drawing
FINGER_NAMES = ["thumb", "index", "middle", "ring", "little"]

DEFECT_TYPES = [
    "hole",
    "pinhole",
    "tear",
    "missing_finger",
    "fused_fingers",
    "stain",
    "dirt",
    "cuff_defect",
    "out_of_tolerance",
]

# Nominal geometry per glove type (mm, degrees). Fingers: thumb, index, middle, ring, little.
GLOVE_TYPES: dict[str, dict] = {
    "exam": dict(
        palm_w=95.0, palm_len=100.0, wrist_w=86.0, cuff_len=60.0, cuff_end_w=92.0,
        finger_w=[21.0, 18.0, 18.5, 18.0, 16.0], finger_len=[58.0, 72.0, 80.0, 75.0, 60.0],
        thumb_base=0.30, thumb_angle=48.0, splay=[-5.0, -1.0, 3.0, 8.0], texture="smooth",
    ),
    "longcuff": dict(
        palm_w=100.0, palm_len=102.0, wrist_w=92.0, cuff_len=150.0, cuff_end_w=135.0,
        finger_w=[22.0, 19.0, 19.5, 19.0, 17.0], finger_len=[60.0, 74.0, 82.0, 77.0, 62.0],
        thumb_base=0.30, thumb_angle=48.0, splay=[-5.0, -1.0, 3.0, 8.0], texture="smooth",
    ),
    "garden": dict(
        palm_w=118.0, palm_len=108.0, wrist_w=110.0, cuff_len=62.0, cuff_end_w=128.0,
        finger_w=[28.0, 24.5, 25.0, 24.5, 22.5], finger_len=[54.0, 66.0, 72.0, 68.0, 55.0],
        thumb_base=0.30, thumb_angle=45.0, splay=[-6.0, -2.0, 3.0, 9.0], texture="knit",
    ),
}

# Product specification used for the dimensional check (also copied into config/config.yaml).
TOLERANCE_MM = {"length": 10.0, "width": 6.0}

STAIN_COLOURS_BGR = [
    (0.20, 0.33, 0.45),  # brown
    (0.25, 0.70, 0.80),  # yellow
    (0.30, 0.30, 0.30),  # grey
    (0.25, 0.25, 0.60),  # reddish
    (0.25, 0.45, 0.25),  # dark green
]


@dataclass
class GloveSpec:
    type: str
    hand: str  # "R" or "L"
    params: dict
    colour_bgr: tuple
    defects: list = field(default_factory=list)  # requested defect types
    scale: float = 1.0


def nominal_dimensions(glove_type: str) -> dict:
    """Length/width of a defect-free, unjittered glove, measured the same way as the GT."""
    spec = GloveSpec(glove_type, "R", copy.deepcopy(GLOVE_TYPES[glove_type]), (0.5, 0.5, 0.5))
    out = render_glove(spec, np.random.default_rng(0), appearance=False)
    return {"length": round(out["length_mm"], 1), "width": round(out["width_mm"], 1)}


def sample_glove(rng: np.random.Generator, glove_type: str, hand: str, colour_bgr, defects: list) -> GloveSpec:
    p = copy.deepcopy(GLOVE_TYPES[glove_type])
    j = lambda v: v * (1.0 + rng.uniform(-0.012, 0.012))  # noqa: E731  production variation
    for k in ("palm_w", "palm_len", "wrist_w", "cuff_len", "cuff_end_w"):
        p[k] = j(p[k])
    p["finger_len"] = [j(v) for v in p["finger_len"]]
    p["finger_w"] = [j(v) for v in p["finger_w"]]
    p["splay"] = [a + rng.uniform(-1.5, 1.5) for a in p["splay"]]
    p["thumb_angle"] += rng.uniform(-3, 3)
    scale = 1.0
    if "out_of_tolerance" in defects:
        scale = rng.uniform(0.90, 0.93) if rng.random() < 0.5 else rng.uniform(1.07, 1.10)
    return GloveSpec(glove_type, hand, p, tuple(colour_bgr), list(defects), scale)


# ----------------------------------------------------------------------------------------------
# geometry helpers
# ----------------------------------------------------------------------------------------------
class _Canvas:
    def __init__(self, xmax: float, ymin: float, ymax: float):
        xmax = math.ceil(xmax * K) / K  # odd width with x=0 on a pixel centre -> exact mirroring
        self.xmax, self.ymin, self.ymax = xmax, ymin, ymax
        self.w = int(round(2 * xmax * K)) + 1
        self.h = int(math.ceil((ymax - ymin) * K))

    def px(self, x, y):
        return ((x + self.xmax) * K, (self.ymax - y) * K)

    def ipx(self, pts):
        f = 1 << SHIFT
        return np.array([[int(round((x + self.xmax) * K * f)), int(round((self.ymax - y) * K * f))] for x, y in pts], np.int32)

    def mm(self, u, v):
        return (u / K - self.xmax, self.ymax - v / K)


def _body_polygon(p, s):
    W, ww, ce, cl, pl = p["palm_w"] * s, p["wrist_w"] * s, p["cuff_end_w"] * s, p["cuff_len"] * s, p["palm_len"] * s
    ypt = cl + pl
    right = [(ce / 2, 0.0), (ww / 2, cl), (W / 2, cl + 0.3 * pl), (W / 2, ypt - 8 * s), (W / 2 - 3 * s, ypt - 1 * s)]
    left = [(-x, y) for x, y in reversed(right)]
    return right + left


def _edge_x(poly, y, side):
    """x of the body outline at height y on side +1 (right) / -1 (left)."""
    pts = [pt for pt in poly if pt[0] * side > 0]
    pts = sorted(pts, key=lambda q: q[1])
    ys = [q[1] for q in pts]
    xs = [q[0] for q in pts]
    return float(np.interp(y, ys, xs))


def _fingers(p, s, finger_len):
    """Returns list of (base, unit_dir, length, width) for the 5 fingers in mm."""
    W, cl, pl = p["palm_w"] * s, p["cuff_len"] * s, p["palm_len"] * s
    ypt = cl + pl
    out = []
    # thumb
    wt = p["finger_w"][0] * s
    a = math.radians(-p["thumb_angle"])
    base = (-W / 2 + 0.35 * wt, cl + p["thumb_base"] * pl)
    out.append((base, (math.sin(a), math.cos(a)), finger_len[0], wt))
    xs = [-0.375, -0.125, 0.125, 0.375]
    for i in range(4):
        a = math.radians(p["splay"][i])
        base = (xs[i] * W, ypt - 3 * s)
        out.append((base, (math.sin(a), math.cos(a)), finger_len[i + 1], p["finger_w"][i + 1] * s))
    return out


def _capsule(img, cv, base, d, length, width, value=255):
    if length <= 0:
        return
    tip = (base[0] + d[0] * (length - width / 2), base[1] + d[1] * (length - width / 2))
    p0, p1 = cv.ipx([base, tip])
    t = max(1, int(round(width * K)))
    cv2.line(img, tuple(p0), tuple(p1), value, t, cv2.LINE_AA, SHIFT)
    r = int(round(width / 2 * K * (1 << SHIFT)))
    cv2.circle(img, tuple(p0), r, value, -1, cv2.LINE_AA, SHIFT)
    cv2.circle(img, tuple(p1), r, value, -1, cv2.LINE_AA, SHIFT)


def _lowfreq(rng, h, w, cells=(6, 8), lo=0.0, hi=1.0):
    small = rng.random((cells[0], cells[1])).astype(np.float32)
    big = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    return np.clip(lo + (hi - lo) * big, min(lo, hi), max(lo, hi))


def _bgr_to_lab(c):
    px = np.array([[c]], np.float32)
    return cv2.cvtColor(px, cv2.COLOR_BGR2LAB)[0, 0]


def delta_e(c1, c2) -> float:
    return float(np.linalg.norm(_bgr_to_lab(c1) - _bgr_to_lab(c2)))


def plate_contrast(c1, c2) -> float:
    """chroma distance + 0.5 |dL| (same measure the inspector's segmentation uses)."""
    a, b = _bgr_to_lab(c1), _bgr_to_lab(c2)
    return float(np.hypot(a[1] - b[1], a[2] - b[2]) + 0.5 * abs(a[0] - b[0]))


# ----------------------------------------------------------------------------------------------
# main renderer
# ----------------------------------------------------------------------------------------------
def render_glove(spec: GloveSpec, rng: np.random.Generator, appearance: bool = True, plate_bgr=None) -> dict:
    """Rasterise one glove with its defects.

    Returns dict with 'alpha' (float32 HxW), 'rgb' (float32 HxWx3, BGR 0..1, if appearance),
    'canvas' (_Canvas), 'gt' (list of defect dicts in local mm), 'length_mm', 'width_mm'.
    """
    p, s = spec.params, spec.scale
    flen = list(p["finger_len"])
    flen = [v * s for v in flen]
    gt: list[dict] = []
    occupied: list[tuple] = []  # (x, y, r) in mm, to keep defects apart

    # --- finger-level defects change the geometry before drawing ---------------------------
    missing_idx = None
    if "missing_finger" in spec.defects:
        missing_idx = int(rng.choice(5, p=[0.12, 0.22, 0.22, 0.22, 0.22]))
        orig = flen[missing_idx]
        frac = rng.uniform(0.4, 1.0)
        new = orig * (1 - frac)
        fw = p["finger_w"][missing_idx] * s
        fingers0 = _fingers(p, s, flen)
        base, d, _, _ = fingers0[missing_idx]
        if new < 0.6 * fw:
            new = 0.0
        flen[missing_idx] = new
        mid = (base[0] + d[0] * orig / 2, base[1] + d[1] * orig / 2)
        gt.append(dict(type="missing_finger", x=mid[0], y=mid[1], finger=FINGER_NAMES[missing_idx],
                       missing_length_mm=round(orig - new, 2), fully_missing=new == 0.0))
        occupied.append((mid[0], mid[1], orig / 2))

    fingers = _fingers(p, s, flen)
    body = _body_polygon(p, s)
    W = p["palm_w"] * s
    thumb_reach = flen[0] + 10 + W / 2 if flen[0] > 0 else W / 2 + 10
    xmax = max(W / 2 + 40 * s, thumb_reach + 15, p["cuff_end_w"] * s / 2 + 15)
    ymax = p["cuff_len"] * s + p["palm_len"] * s + max(flen) + 15
    cv = _Canvas(xmax, -10.0, ymax)
    mask = np.zeros((cv.h, cv.w), np.uint8)
    cv2.fillPoly(mask, [cv.ipx(body)], 255, cv2.LINE_AA, SHIFT)
    for (base, d, L, w) in fingers:
        _capsule(mask, cv, base, d, L, w)

    if "fused_fingers" in spec.defects:
        pairs = [(1, 2), (2, 3), (3, 4)]
        pairs = [pr for pr in pairs if missing_idx not in pr]
        i, j = pairs[int(rng.integers(len(pairs)))]
        f = rng.uniform(0.55, 0.8)
        (bi, di, Li, _), (bj, dj, Lj, _) = fingers[i], fingers[j]
        h = f * min(Li, Lj)
        poly = [bi, (bi[0] + di[0] * h, bi[1] + di[1] * h), (bj[0] + dj[0] * h, bj[1] + dj[1] * h), bj]
        cv2.fillPoly(mask, [cv.ipx(poly)], 255, cv2.LINE_AA, SHIFT)
        cx = sum(q[0] for q in poly) / 4
        cy = sum(q[1] for q in poly) / 4
        gt.append(dict(type="fused_fingers", x=cx, y=cy, fingers=f"{FINGER_NAMES[i]}-{FINGER_NAMES[j]}",
                       web_height_mm=round(h, 2)))
        occupied.append((cx, cy, h / 2))

    def body_mask():
        m = np.zeros_like(mask)
        cv2.fillPoly(m, [cv.ipx(body)], 255, cv2.LINE_AA, SHIFT)
        return m

    def pick_center(region_mask, r_mm, margin_mm, tries=200):
        dist = cv2.distanceTransform((region_mask > 127).astype(np.uint8), cv2.DIST_L2, 5) / K
        ys, xs = np.nonzero(dist >= r_mm + margin_mm)
        if len(xs) == 0:
            return None
        for _ in range(tries):
            k = int(rng.integers(len(xs)))
            x, y = cv.mm(xs[k], ys[k])
            if all(math.hypot(x - ox, y - oy) > r_mm + orr + 8 for ox, oy, orr in occupied):
                return x, y
        return None

    # --- edge defects -------------------------------------------------------------------------
    if "tear" in spec.defects:
        cl, pl = p["cuff_len"] * s, p["palm_len"] * s
        ypt = cl + pl
        thumb_side = rng.random() < 0.35
        if thumb_side:
            side = -1
            y_hi = cl + p["thumb_base"] * pl - (p["finger_w"][0] * s / 2 + 6)
        else:
            side = 1
            y_hi = ypt - 25 * s
        for _ in range(50):
            y = rng.uniform(18, y_hi)
            if all(math.hypot(side * W / 2 - ox, y - oy) > orr + 20 for ox, oy, orr in occupied):
                break
        ex = _edge_x(body, y, side)
        m = rng.uniform(3.0, 8.0)
        d = rng.uniform(6.0, 15.0)
        t = math.radians(rng.uniform(-20, 20))
        inward = (-side * math.cos(t), math.sin(t))
        apex = (ex + inward[0] * d, y + inward[1] * d)
        poly = [(ex + side * 3, y - m / 2), apex, (ex + side * 3, y + m / 2)]
        cv2.fillPoly(mask, [cv.ipx(poly)], 0, cv2.LINE_AA, SHIFT)
        depth = d * math.cos(t)
        cx, cy = ex - side * depth / 2, y + inward[1] * d / 2
        gt.append(dict(type="tear", x=cx, y=cy, depth_mm=round(depth, 2), length_mm=round(d, 2),
                       side="thumb" if thumb_side else "little"))
        occupied.append((cx, cy, d / 2))

    if "cuff_defect" in spec.defects:
        ce = p["cuff_end_w"] * s
        if rng.random() < 0.5:
            r = rng.uniform(6.0, 14.0)
            x = rng.uniform(-0.3, 0.3) * ce
            c = cv.ipx([(x, 0.0)])[0]
            cv2.circle(mask, tuple(c), int(r * K * (1 << SHIFT)), 0, -1, cv2.LINE_AA, SHIFT)
            gt.append(dict(type="cuff_defect", x=x, y=r / 2, kind="bite", max_dev_mm=round(r, 2)))
            occupied.append((x, r / 2, r))
        else:
            span = rng.uniform(0.4, 0.7) * ce
            xa = rng.uniform(-ce / 2 + 8, ce / 2 - 8 - span)
            n = int(rng.integers(3, 7))
            amp = rng.uniform(4.0, 9.0)
            pts = [(xa, -3.0)]
            peaks = []
            for k in range(n):
                xk = xa + span * (k + 0.5) / n
                hk = amp * rng.uniform(0.6, 1.0)
                peaks.append(hk)
                pts.append((xk, hk))
                pts.append((xa + span * (k + 1) / n, 0.3))
            pts[-1] = (xa + span, -3.0)
            cv2.fillPoly(mask, [cv.ipx(pts)], 0, cv2.LINE_AA, SHIFT)
            gt.append(dict(type="cuff_defect", x=xa + span / 2, y=amp / 2, kind="jagged",
                           max_dev_mm=round(max(peaks), 2)))
            occupied.append((xa + span / 2, amp / 2, span / 2))

    # --- holes ----------------------------------------------------------------------------------
    if "hole" in spec.defects:
        dia = rng.uniform(4.5, 15.0)
        r = dia / 2
        region = cv2.bitwise_and(body_mask(), mask)
        c = pick_center(region, r * 1.2, 6.0)
        if c is not None:
            n = 18
            noise = cv2.GaussianBlur(rng.normal(0, 1, (n, 1)).astype(np.float32), (1, 5), 1.2).ravel()
            ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
            rr = r * (1 + 0.18 * noise / (np.abs(noise).max() + 1e-6))
            poly = [(c[0] + rr[k] * math.cos(ang[k]), c[1] + rr[k] * math.sin(ang[k])) for k in range(n)]
            before = mask.astype(np.float32).sum()
            cv2.fillPoly(mask, [cv.ipx(poly)], 0, cv2.LINE_AA, SHIFT)
            area = (before - mask.astype(np.float32).sum()) / 255.0 / K**2
            gt.append(dict(type="hole", x=c[0], y=c[1], diameter_mm=round(math.sqrt(4 * area / math.pi), 2),
                           area_mm2=round(area, 2)))
            occupied.append((c[0], c[1], r * 1.2))

    if "pinhole" in spec.defects:
        dia = rng.uniform(1.8, 3.2)
        c = pick_center(mask, dia / 2, 5.0)
        if c is not None:
            q = cv.ipx([c])[0]
            cv2.circle(mask, tuple(q), int(dia / 2 * K * (1 << SHIFT)), 0, -1, cv2.LINE_AA, SHIFT)
            gt.append(dict(type="pinhole", x=c[0], y=c[1], diameter_mm=round(dia, 2)))
            occupied.append((c[0], c[1], dia / 2))

    alpha = mask.astype(np.float32) / 255.0

    # --- ground-truth dimensions -----------------------------------------------------------------
    ys = np.nonzero(mask.max(axis=1) > 127)[0]
    length_mm = (ys.max() - ys.min() + 1) / K
    width_mm = W

    result = dict(alpha=alpha, canvas=cv, gt=gt, length_mm=float(length_mm), width_mm=float(width_mm))
    if not appearance:
        return _mirror(result, spec.hand)

    # --- appearance ------------------------------------------------------------------------------
    h, w = mask.shape
    base = np.array(spec.colour_bgr, np.float32)
    shade = _lowfreq(rng, h, w, (5, 4), 0.88, 1.06)
    creases = np.ones((h, w), np.float32)
    for _ in range(int(rng.integers(6, 13))):
        x0, y0 = rng.uniform(0, w), rng.uniform(0, h)
        ang = rng.uniform(0, np.pi)
        ln = rng.uniform(20, 70) * K
        x1, y1 = x0 + ln * math.cos(ang), y0 + ln * math.sin(ang)
        cv2.line(creases, (int(x0), int(y0)), (int(x1), int(y1)), float(rng.uniform(0.86, 0.94)),
                 int(rng.integers(2, 6)), cv2.LINE_AA)
    creases = cv2.GaussianBlur(creases, (0, 0), 2.0)
    shade = shade * creases
    if p["texture"] == "knit":
        knit = cv2.GaussianBlur(rng.normal(0, 0.06, (h, w)).astype(np.float32), (0, 0), 1.2)
        shade = shade * (1 + knit * 2.0)
    edge = cv2.distanceTransform((mask > 127).astype(np.uint8), cv2.DIST_L2, 5) / K
    shade = shade * np.where(edge < 2.0, 0.93, 1.0).astype(np.float32)
    rgb = base[None, None, :] * shade[:, :, None]

    if "stain" in spec.defects:
        D = rng.uniform(8.0, 24.0)
        asp = rng.uniform(1.0, 1.8)
        a, b = D / 2 * math.sqrt(asp), D / 2 / math.sqrt(asp)
        c = pick_center(mask, a, 5.0)
        if c is not None:
            # A stain that looks like the plate would be indistinguishable from a hole for any camera,
            # so stain colours must differ from the glove AND from the plate.
            best = None
            for col in rng.permutation(len(STAIN_COLOURS_BGR)):
                strength = rng.uniform(0.6, 0.85)
                sc = tuple(np.array(spec.colour_bgr) * (1 - strength) + np.array(STAIN_COLOURS_BGR[col]) * strength)
                de = delta_e(spec.colour_bgr, sc)
                pc = plate_contrast(sc, plate_bgr) if plate_bgr is not None else 99.0
                score = min(de, pc + 5)
                if best is None or score > best[2]:
                    best = (de, sc, score)
                if de >= 18 and pc >= 20:
                    break
            sm = np.zeros((h, w), np.float32)
            q = cv.px(*c)
            cv2.ellipse(sm, (int(q[0]), int(q[1])), (int(a * K), int(b * K)), float(rng.uniform(0, 180)), 0, 360, 1.0, -1,
                        cv2.LINE_AA)
            hard = sm.copy()
            sm = cv2.GaussianBlur(sm, (0, 0), 0.6 * K) * (mask > 127)
            target = np.array(best[1], np.float32)
            rgb = rgb * (1 - sm[:, :, None]) + (target[None, None, :] * shade[:, :, None]) * sm[:, :, None]
            area = float(hard.sum()) / K**2
            gt.append(dict(type="stain", x=c[0], y=c[1], diameter_mm=round(math.sqrt(4 * area / math.pi), 2),
                           area_mm2=round(area, 2), delta_e=round(best[0], 1)))
            occupied.append((c[0], c[1], a))

    if "dirt" in spec.defects:
        for _ in range(int(rng.integers(1, 4))):
            dia = rng.uniform(1.5, 4.0)
            c = pick_center(mask, dia / 2, 4.0)
            if c is None:
                continue
            q = cv.px(*c)
            sm = np.zeros((h, w), np.float32)
            cv2.circle(sm, (int(q[0]), int(q[1])), max(1, int(dia / 2 * K)), 1.0, -1, cv2.LINE_AA)
            sm = cv2.GaussianBlur(sm, (0, 0), 0.25 * K)
            col = np.array([rng.uniform(0.08, 0.16)] * 3, np.float32) * np.array([0.9, 1.0, 1.1], np.float32)
            rgb = rgb * (1 - sm[:, :, None]) + col[None, None, :] * sm[:, :, None]
            gt.append(dict(type="dirt", x=c[0], y=c[1], diameter_mm=round(dia, 2)))
            occupied.append((c[0], c[1], dia / 2 + 2))

    if "out_of_tolerance" in spec.defects:
        gt.append(dict(type="out_of_tolerance", x=0.0, y=float(length_mm) / 2, scale=round(s, 3)))

    result["rgb"] = np.clip(rgb, 0, 1).astype(np.float32)
    return _mirror(result, spec.hand)


def _mirror(result: dict, hand: str) -> dict:
    if hand == "R":
        return result
    result["alpha"] = np.ascontiguousarray(result["alpha"][:, ::-1])
    if "rgb" in result:
        result["rgb"] = np.ascontiguousarray(result["rgb"][:, ::-1])
    for g in result["gt"]:
        g["x"] = -g["x"]
    return result
