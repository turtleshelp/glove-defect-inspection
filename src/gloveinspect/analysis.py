"""Per-glove analysis: pose normalisation, measurements, orientation and all defect detectors.

Everything works on the rectified image (fixed px/mm). The glove is rotated into a normalised frame
("N-frame"): principal axis vertical, fingers up, cuff down. In that frame:

* length            = vertical extent; width = palm width 15 mm below the finger valleys
* top/bottom        = which end of the principal axis carries the fingers (rows with >= 3 runs)
* left/right hand   = side on which the thumb sticks out (back of the hand facing the camera)
* fingers           = peaks of the top profile; thumb = lateral peak; valleys between them
* defects           = holes/pinholes (plate visible inside), tears (notches in the side outline),
                      missing/short and fused fingers (finger peaks and valleys vs. learned nominal
                      ratios), stains/dirt (local colour deviation), cuff defects (cuff-end profile),
                      out-of-tolerance (length/width vs. spec)
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import find_peaks

FINGERS = ["index", "middle", "ring", "little"]

DEFAULT_GOLDEN = dict(
    finger_len_ratio=[0.76, 0.84, 0.79, 0.63],
    tip_s_ratio=[-0.44, -0.13, 0.17, 0.46],
    thumb_asym=0.42,
    thumb_tip=[-0.90, 0.45],
    valley_elev_mm=2.0,
    palm_top_ratio=0.34,
)


@dataclass
class Defect:
    type: str
    x: float            # rectified image px
    y: float
    gx_mm: float        # position in the glove frame (mm from centre line / from fingertip end)
    gy_mm: float
    size_mm: float      # main measurement (diameter, depth, missing length, ...)
    measures: dict = field(default_factory=dict)


@dataclass
class GloveObservation:
    contour: np.ndarray
    center: tuple
    type: str
    type_conf: float
    type_probs: dict
    hand: str                 # "R", "L" or "?"
    direction: str            # "fingers_up" / "fingers_down"
    length_mm: float
    width_mm: float
    features: dict
    defects: list
    tip_pt: tuple             # rectified px, fingertip end of the axis
    cuff_pt: tuple
    thumb_pt: tuple | None
    stats: dict = field(default_factory=dict)  # raw shape statistics (used to learn golden values)


# ----------------------------------------------------------------------------------------------
def _rot(angle_deg, center, shape, pad=6):
    h, w = shape[:2]
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], np.float64)
    tc = corners @ M.T
    lo = tc.min(axis=0)
    hi = tc.max(axis=0)
    M[:, 2] += -lo + pad
    size = (int(math.ceil(hi[0] - lo[0])) + 2 * pad, int(math.ceil(hi[1] - lo[1])) + 2 * pad)
    return M, size


def _warp_mask(F, M, size):
    return cv2.warpAffine(F.astype(np.float32), M, size, flags=cv2.INTER_LINEAR) > 0.5


def _edges(Fb):
    any_r = Fb.any(axis=1)
    xL = np.where(any_r, Fb.argmax(axis=1), -1)
    xR = np.where(any_r, Fb.shape[1] - 1 - Fb[:, ::-1].argmax(axis=1), -1)
    return any_r, xL, xR


def _robust_line(x, y, tol, init_mask=None, iters=3):
    m = np.ones(len(x), bool) if init_mask is None else init_mask.copy()
    if m.sum() < 2:
        m[:] = True
    p = np.polyfit(x[m], y[m], 1)
    for _ in range(iters):
        r = y - np.polyval(p, x)
        m2 = np.abs(r) < tol
        if m2.sum() < max(2, 0.2 * len(x)):
            break
        m = m2
        p = np.polyfit(x[m], y[m], 1)
    return p


def shape_features(Fb, y_top, y_bot, xL, xR, ppm):
    L = y_bot - y_top + 1
    widths = np.where(xL >= 0, xR - xL + 1, 0).astype(np.float64)
    area = float(Fb.sum())
    cnts, _ = cv2.findContours(Fb.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(cnts, key=cv2.contourArea)
    hull_area = cv2.contourArea(cv2.convexHull(cnt)) + 1e-6
    maxw = widths.max()
    f = dict(
        length_mm=L / ppm,
        area_mm2=area / ppm ** 2,
        max_width_mm=maxw / ppm,
        aspect=L / maxw,
        solidity=area / hull_area,
        extent=area / (L * maxw),
    )
    for frac in (0.5, 0.65, 0.8, 0.95):
        y = int(y_top + frac * (L - 1))
        f[f"w{int(frac * 100)}_mm"] = float(np.median(widths[max(y_top, y - 2):y + 3])) / ppm
    hu = cv2.HuMoments(cv2.moments(Fb.astype(np.uint8), binaryImage=True)).ravel()
    for i in range(3):
        f[f"hu{i + 1}"] = float(-np.sign(hu[i]) * np.log10(abs(hu[i]) + 1e-30))
    return f


# ----------------------------------------------------------------------------------------------
def analyse_glove(seg, contour, ppm: float, cfg: dict, classifier, golden: dict) -> GloveObservation:
    dc = cfg["defects"]
    specs, tol = cfg["glove_types"], cfg["tolerance"]
    H, W = seg.mask.shape
    bx, by, bw, bh = cv2.boundingRect(contour)
    pad = 30
    x0, y0 = max(0, bx - pad), max(0, by - pad)
    x1, y1 = min(W, bx + bw + pad), min(H, by + bh + pad)
    F = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.drawContours(F, [contour - [x0, y0]], -1, 1, -1)
    de_c = seg.de[y0:y1, x0:x1]
    lab_c = seg.lab[y0:y1, x0:x1]

    # ---- pose normalisation --------------------------------------------------------------------
    ys, xs = np.nonzero(F)
    mean = np.array([xs.mean(), ys.mean()])
    cov = np.cov(np.stack([xs - mean[0], ys - mean[1]]))
    evals, evecs = np.linalg.eigh(cov)
    vx, vy = evecs[:, np.argmax(evals)]
    ctr = (float(mean[0]), float(mean[1]))
    ang = math.degrees(math.atan2(-vx, vy))
    M, size = _rot(ang, ctr, F.shape)
    Fb = _warp_mask(F, M, size)
    any_r, _, _ = _edges(Fb)
    rows = np.nonzero(any_r)[0]
    yt, yb = rows[0], rows[-1]
    starts = (np.diff(Fb.astype(np.int8), axis=1, prepend=0) == 1).sum(axis=1)
    mid = (yt + yb) // 2
    if (starts[mid:yb + 1] >= 3).sum() > (starts[yt:mid] >= 3).sum():
        ang += 180.0
        M, size = _rot(ang, ctr, F.shape)
        Fb = _warp_mask(F, M, size)
    any_r, xL, xR = _edges(Fb)
    rows = np.nonzero(any_r)[0]
    yt, yb = rows[0], rows[-1]
    Lp = yb - yt + 1
    r0, r1 = int(yb - 0.30 * Lp), int(yb - 3 * ppm)
    rr = np.arange(r0, r1)
    rr = rr[any_r[rr]]
    if len(rr) > 10:
        midx = (xL[rr] + xR[rr]) / 2.0
        slope = _robust_line(rr.astype(float), midx, 2.0)[0]
        ang += math.degrees(math.atan(-slope))
        M, size = _rot(ang, ctr, F.shape)
        Fb = _warp_mask(F, M, size)
    de = cv2.warpAffine(de_c, M, size, flags=cv2.INTER_LINEAR)
    plate_L = cv2.warpAffine(seg.plate_L[y0:y1, x0:x1], M, size, flags=cv2.INTER_LINEAR) if seg.plate_L is not None else None
    lab = cv2.warpAffine(lab_c, M, size, flags=cv2.INTER_LINEAR)
    Minv = cv2.invertAffineTransform(M)

    def to_rect(x, y):
        p = Minv @ np.array([x, y, 1.0])
        return float(p[0] + x0), float(p[1] + y0)

    any_r, xL, xR = _edges(Fb)
    rows = np.nonzero(any_r)[0]
    yt, yb = int(rows[0]), int(rows[-1])
    Lp = yb - yt + 1
    widths = np.where(xL >= 0, xR - xL + 1, 0)
    rr = np.arange(int(yb - 0.30 * Lp), int(yb - 3 * ppm))
    rr = rr[any_r[rr]]
    c = float(np.median((xL[rr] + xR[rr]) / 2.0)) if len(rr) else float(np.nonzero(Fb.any(0))[0].mean())
    ext_left = c - xL[any_r].min()
    ext_right = xR[any_r].max() - c

    feats = shape_features(Fb, yt, yb, xL, xR, ppm)
    gtype, conf, probs = classifier.predict(feats)
    g = dict(DEFAULT_GOLDEN)
    g.update(golden.get(gtype, {}))

    # ---- fingers -------------------------------------------------------------------------------
    cols_any = Fb.any(axis=0)
    top = np.where(cols_any, Fb.argmax(axis=0), yb).astype(np.float64)
    topf = median_filter(top, 5, mode="nearest")
    prom = dc["finger_prominence"] * ppm
    pk, props = find_peaks(-topf, prominence=prom, distance=max(1, int(8 * ppm)))
    proms = props["prominences"]
    thumb_left = ext_left > ext_right
    sgn = 1.0 if thumb_left else -1.0
    s_of = lambda x: (x - c) * sgn  # noqa: E731  lateral coordinate, increasing from thumb side
    x_of = lambda s: c + s * sgn  # noqa: E731
    ext_thumb, ext_other = (ext_left, ext_right) if thumb_left else (ext_right, ext_left)
    # thumb tip = outermost glove point on the thumb side
    if thumb_left:
        ty = int(np.argmin(np.where(any_r, xL, 10 ** 6)))
        tx = float(xL[ty])
    else:
        ty = int(np.argmax(np.where(any_r, xR, -1)))
        tx = float(xR[ty])

    thumb_pk = None
    cand = [i for i in range(len(pk)) if s_of(pk[i]) < -0.9 * ext_other]
    if cand:
        thumb_pk = min(cand, key=lambda i: s_of(pk[i]))
    fidx = [i for i in range(len(pk)) if i != thumb_pk]
    if len(fidx) > 4:
        fidx = sorted(sorted(fidx, key=lambda i: -proms[i])[:4])
    fidx = sorted(fidx, key=lambda i: s_of(pk[i]))
    valleys = []
    for a, b in zip(fidx[:-1], fidx[1:]):
        lo, hi = sorted((pk[a], pk[b]))
        seg_ = topf[lo:hi + 1]
        k = int(np.argmax(seg_))
        valleys.append((a, b, lo + k, float(seg_[k])))
    if valleys:
        palm_top = max(v[3] for v in valleys)
    else:
        palm_top = yt + g["palm_top_ratio"] * Lp

    yw = int(round(palm_top + 15 * ppm))
    width_px = None
    if yt <= yw <= yb:
        row = Fb[yw].astype(np.int8)
        d = np.diff(np.concatenate([[0], row, [0]]))
        st, en = np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]
        if len(st):
            dist = [0 if s0 <= c < e0 else min(abs(c - s0), abs(c - e0)) for s0, e0 in zip(st, en)]
            k = int(np.argmin(dist))
            width_px = en[k] - st[k]
    if width_px is None:
        width_px = widths.max()
    width_mm = width_px / ppm
    length_mm = Lp / ppm

    defects: list[Defect] = []

    def add(t, x, y, size, **meas):
        meas = {k: (v.item() if isinstance(v, np.generic) else v) for k, v in meas.items()}
        rx, ry = to_rect(x, y)
        defects.append(Defect(t, rx, ry, (x - c) / ppm, (y - yt) / ppm, float(size), meas))

    # finger slots
    s_exp = np.array(g["tip_s_ratio"]) * width_px
    len_exp = np.array(g["finger_len_ratio"]) * width_mm
    k = len(fidx)
    best, best_cost = (), float("inf")
    for combo in itertools.combinations(range(4), min(k, 4)):
        cost = sum(abs(s_of(pk[fidx[i]]) - s_exp[j]) for i, j in enumerate(combo))
        if cost < best_cost:
            best, best_cost = combo, cost
    slot_of = {fidx[i]: j for i, j in enumerate(best)}
    finger_lengths = {}
    for j in range(4):
        idxs = [i for i, sj in slot_of.items() if sj == j]
        if idxs:
            i = idxs[0]
            Lf = (palm_top - topf[pk[i]]) / ppm
            finger_lengths[FINGERS[j]] = Lf
            if Lf < dc["short_finger_ratio"] * len_exp[j]:
                miss = len_exp[j] - Lf
                add("missing_finger", pk[i], topf[pk[i]] + miss * ppm / 2, miss, finger=FINGERS[j],
                    measured_mm=round(Lf, 1), expected_mm=round(len_exp[j], 1))
        else:
            add("missing_finger", x_of(s_exp[j]), palm_top - len_exp[j] * ppm / 2, len_exp[j], finger=FINGERS[j],
                measured_mm=0.0, expected_mm=round(len_exp[j], 1))

    thumb_asym = (ext_thumb - ext_other) / width_px
    tt = g["thumb_tip"]
    if thumb_asym < dc["thumb_ratio"] * g["thumb_asym"]:
        sx = (tt[0] - 0.5) / 2 * width_px
        add("missing_finger", x_of(sx), palm_top + tt[1] * width_px, (1 - thumb_asym / g["thumb_asym"]) * 60,
            finger="thumb", protrusion_ratio=round(thumb_asym / g["thumb_asym"], 2))
    hand = ("R" if thumb_left else "L") if thumb_asym > 0.35 * g["thumb_asym"] else "?"

    valley_elev = []
    for a, b, vx_, vy_ in valleys:
        ja, jb = slot_of.get(a), slot_of.get(b)
        elev = (palm_top - vy_) / ppm
        valley_elev.append(elev)
        if ja is not None and jb is not None and jb - ja == 1 and elev - g["valley_elev_mm"] >= dc["fused_min_elevation"]:
            add("fused_fingers", vx_, (vy_ + palm_top) / 2, elev, fingers=f"{FINGERS[ja]}-{FINGERS[jb]}")

    # ---- tears (notches in the side outline below the finger zone) ---------------------------
    wpx, gpx = int(dc["tear_window"] * ppm), int(dc["tear_gap"] * ppm)
    ya, yz = int(palm_top + 8 * ppm), int(yb - 4 * ppm)
    for side in ("left", "right"):
        e = xL.astype(np.float64) if side == "left" else -xR.astype(np.float64)  # larger = more inward
        dev = np.zeros(len(e))
        for y in range(max(ya, yt), max(ya, yz)):
            up = e[max(yt, y - wpx):max(yt, y - gpx)]
            dn = e[min(yb, y + gpx):min(yb, y + wpx)]
            if len(up) < 3 or len(dn) < 3:
                continue
            dev[y] = e[y] - max(np.median(up), np.median(dn))
        hit = dev >= dc["tear_min_depth"] * ppm
        lab_runs, n = _runs(hit)
        for (s0, e0) in lab_runs:
            if (e0 - s0) / ppm > 25:
                continue
            yk = s0 + int(np.argmax(dev[s0:e0]))
            depth = dev[yk]
            xe = xL[yk] if side == "left" else xR[yk]
            xm = xe - depth / 2 if side == "left" else xe + depth / 2
            add("tear", xm, yk, depth / ppm, depth_mm=round(depth / ppm, 2), length_mm=round((e0 - s0) / ppm, 1))

    # ---- cuff end ------------------------------------------------------------------------------
    band = int(25 * ppm)
    cols = np.nonzero(cols_any)[0]
    bot = np.array([np.nonzero(Fb[:, x])[0].max() for x in cols])
    sel = bot >= yb - band
    cx_, cb_ = cols[sel], bot[sel].astype(np.float64)
    if len(cx_) > 20:
        n = len(cx_)
        keep = slice(int(0.05 * n), int(0.95 * n))
        cx_, cb_ = cx_[keep], cb_[keep]
        init = cb_ >= np.percentile(cb_, 60)
        p = _robust_line(cx_.astype(float), cb_, 1.5 * ppm, init)
        devc = np.polyval(p, cx_) - cb_
        kk = int(np.argmax(devc))
        if devc[kk] >= dc["cuff_min_dev"] * ppm:
            hit = devc >= 0.5 * dc["cuff_min_dev"] * ppm
            runs, _ = _runs(hit)
            for (s0, e0) in runs:
                k2 = s0 + int(np.argmax(devc[s0:e0]))
                if devc[k2] >= dc["cuff_min_dev"] * ppm:
                    add("cuff_defect", cx_[k2], cb_[k2] + devc[k2] / 2, devc[k2] / ppm,
                        max_dev_mm=round(devc[k2] / ppm, 2), span_mm=round((cx_[e0 - 1] - cx_[s0]) / ppm, 1))

    # ---- holes and pinholes ----------------------------------------------------------------------
    # A hole shows the plate: its colour distance to the learned plate drops well below that of the
    # surrounding glove material. The reference is local (normalised Gaussian average of the glove
    # around it), so smooth shadows across a light glove do not look like holes.
    sig = dc["local_ref_window"] * ppm / 4
    inner = cv2.erode(Fb.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    # how far plate pixels next to the glove drift from the learned plate (noise, shadows) -> a hole
    # must look at least that much like the plate
    ring = cv2.dilate(Fb.astype(np.uint8), np.ones((int(12 * ppm) | 1,) * 2, np.uint8)).astype(bool) & ~cv2.dilate(
        Fb.astype(np.uint8), np.ones((int(3 * ppm) | 1,) * 2, np.uint8)).astype(bool)
    plate_de = float(np.percentile(de[ring], 90)) if ring.any() else 5.0
    plate_med = float(np.median(de[ring])) if ring.any() else 2.0
    abs_thr = max(dc["hole_abs_min"], dc["hole_abs_factor"] * plate_de)
    rel = dc["hole_bg_fraction"]
    de_ref = _nconv(de, inner, sig)
    cand = (de < rel * de_ref) & inner
    w1 = inner & ~cv2.dilate(cand.astype(np.uint8), np.ones((int(3 * ppm) | 1,) * 2, np.uint8)).astype(bool)
    de_ref = _nconv(de, w1, sig)
    holes = (de < rel * de_ref) & inner
    # sub-pixel "plateness" (0 = glove, 1 = plate) for an unbiased area of small, blurred holes
    plateness = np.clip((de_ref - de) / np.maximum(de_ref - plate_med, 1.0), 0, 1)
    plateness[plateness < 0.15] = 0
    nlab, lbl, st, cen = cv2.connectedComponentsWithStats(holes.astype(np.uint8), connectivity=8)
    hole_mask = np.zeros_like(holes)
    # components within ~3 mm of the outline are notches/tears, not holes
    outside = cv2.dilate((~Fb).astype(np.uint8), np.ones((int(6 * ppm) | 1,) * 2, np.uint8)).astype(bool)
    k5 = np.ones((5, 5), np.uint8)
    for i in range(1, nlab):
        x_, y_, w_, h_ = st[i, :4]
        sl = (slice(max(0, y_ - 3), y_ + h_ + 3), slice(max(0, x_ - 3), x_ + w_ + 3))
        comp = np.zeros_like(holes)
        comp[sl] = lbl[sl] == i
        if (comp[sl] & outside[sl]).any():
            continue
        region = cv2.dilate(comp[sl].astype(np.uint8), k5).astype(bool) & inner[sl]
        a = float(plateness[sl][region].sum()) / ppm ** 2
        if a < dc["min_hole_area_mm2"]:
            continue
        dia = math.sqrt(4 * a / math.pi)
        # a larger opening must really look like the plate (a plate-coloured stain must not count);
        # pinholes are too small to reach the plate colour after camera blur, so only the relative test applies
        if dia >= dc["pinhole_max_diameter"] and float(de[sl][comp[sl]].min()) >= abs_thr:
            continue
        # a dark particle is darker than both the plate and the glove around it; a hole never is
        if plate_L is not None:
            Lc = lab[:, :, 0][sl]
            glove_L = float(np.median(Lc[inner[sl] & ~region])) if (inner[sl] & ~region).any() else 100.0
            if float(Lc[region].min()) < min(glove_L, float(np.median(plate_L[sl][region]))) - 12:
                continue
        hole_mask |= comp
        t = "pinhole" if dia < dc["pinhole_max_diameter"] else "hole"
        add(t, cen[i][0], cen[i][1], dia, diameter_mm=round(dia, 2), area_mm2=round(a, 1))

    # ---- stains and dirt (local colour deviation) ----------------------------------------------
    eb = max(1, int(round(dc["edge_band"] * ppm)))
    interior = cv2.erode(Fb.astype(np.uint8), np.ones((2 * eb + 1, 2 * eb + 1), np.uint8)).astype(bool)
    interior &= ~cv2.dilate(hole_mask.astype(np.uint8), np.ones((int(4 * ppm) + 1,) * 2, np.uint8)).astype(bool)
    if interior.sum() > 100:
        Lr, ar, br = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
        as_, bs_ = cv2.GaussianBlur(ar, (0, 0), 1.0), cv2.GaussianBlur(br, (0, 0), 1.0)

        def deviation(weight):
            Lref, aref, bref = (_nconv(ch, weight, sig) for ch in (Lr, ar, br))
            return Lr - Lref, np.hypot(as_ - aref, bs_ - bref)

        def candidates(dL, dC):
            return interior & ((dC >= dc["stain_chroma"]) | (dL <= -dc["stain_dl"]) | (dL <= -dc["dirt_dl"] * 0.6))

        dL, dC = deviation(interior)
        c0 = candidates(dL, dC)
        w1 = interior & ~cv2.dilate(c0.astype(np.uint8), np.ones((int(4 * ppm) | 1,) * 2, np.uint8)).astype(bool)
        dL, dC = deviation(w1)  # second pass: reference without the defects themselves
        cand = candidates(dL, dC)
        cand = cv2.morphologyEx(cand.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        nlab, lbl, st, cen = cv2.connectedComponentsWithStats(cand, connectivity=8)
        for i in range(1, nlab):
            a = st[i, cv2.CC_STAT_AREA] / ppm ** 2
            if a < dc["min_spot_area_mm2"]:
                continue
            comp = lbl == i
            dia = math.sqrt(4 * a / math.pi)
            dmin = float(dL[comp].min())
            if dia <= dc["dirt_max_diameter"] and dmin <= -dc["dirt_dl"]:
                add("dirt", cen[i][0], cen[i][1], dia, diameter_mm=round(dia, 2))
            elif dia >= dc["stain_min_diameter"]:
                add("stain", cen[i][0], cen[i][1], dia, diameter_mm=round(dia, 2), area_mm2=round(a, 1),
                    delta_e=round(float(np.mean(np.hypot(dC[comp], dL[comp]))), 1))

    # ---- dimensions --------------------------------------------------------------------------------
    sp = specs.get(gtype)
    if sp is not None:
        dl, dw = length_mm - sp["length"], width_mm - sp["width"]
        if abs(dl) > tol["length"] or abs(dw) > tol["width"]:
            size = max(abs(dl) - tol["length"], abs(dw) - tol["width"])
            add("out_of_tolerance", c, (yt + yb) / 2, size, length_dev_mm=round(dl, 1), width_dev_mm=round(dw, 1))

    # ---- orientation & output -----------------------------------------------------------------------
    tip = to_rect(c, yt)
    cuff = to_rect(c, yb)
    direction = "fingers_up" if tip[1] < cuff[1] else "fingers_down"
    thumb_pt = to_rect(tx, ty) if hand != "?" else None
    mm = cv2.moments(contour)
    center = (mm["m10"] / mm["m00"], mm["m01"] / mm["m00"])
    stats = dict(
        width_px=float(width_px), palm_top_ratio=float((palm_top - yt) / Lp), thumb_asym=float(thumb_asym),
        finger_s=[float(s_of(pk[i]) / width_px) for i in fidx],
        finger_len=[float((palm_top - topf[pk[i]]) / ppm / width_mm) for i in fidx],
        valley_elev=sorted(valley_elev)[:-1] if len(valley_elev) > 1 else [],
        thumb_tip=[float(s_of(tx) / width_px), float((ty - palm_top) / width_px)],
        n_finger_peaks=len(fidx),
    )
    return GloveObservation(contour=contour, center=center, type=gtype, type_conf=conf, type_probs=probs, hand=hand,
                            direction=direction, length_mm=float(length_mm), width_mm=float(width_mm), features=feats,
                            defects=defects, tip_pt=tip, cuff_pt=cuff, thumb_pt=thumb_pt, stats=stats)


def _nconv(ch, weight, sigma):
    """Normalised convolution: Gaussian-weighted local mean of ch over the pixels where weight is set."""
    w = weight.astype(np.float32)
    num = cv2.GaussianBlur(ch.astype(np.float32) * w, (0, 0), sigma)
    den = cv2.GaussianBlur(w, (0, 0), sigma)
    return num / np.maximum(den, 1e-4)


def _runs(mask):
    m = np.asarray(mask, bool).astype(np.int8)
    d = np.diff(np.concatenate([[0], m, [0]]))
    st, en = np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]
    return list(zip(st, en)), len(st)
