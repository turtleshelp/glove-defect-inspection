"""Drawing of inspection results on the rectified frame."""
from __future__ import annotations

import cv2
import numpy as np

DECISION_COLOURS = {"PASS": (60, 200, 60), "REWORK": (0, 165, 255), "REJECT": (40, 40, 230)}
DEFECT_COLOURS = {
    "hole": (40, 40, 255), "pinhole": (180, 60, 255), "tear": (0, 0, 200), "missing_finger": (255, 0, 255),
    "fused_fingers": (255, 120, 0), "stain": (0, 200, 255), "dirt": (220, 220, 220), "cuff_defect": (255, 255, 0),
    "out_of_tolerance": (200, 200, 200),
}
SHORT = {"hole": "HOLE", "pinhole": "PINHOLE", "tear": "TEAR", "missing_finger": "MISSING FINGER",
         "fused_fingers": "FUSED", "stain": "STAIN", "dirt": "DIRT", "cuff_defect": "CUFF", "out_of_tolerance": "SIZE"}


def _text(img, txt, org, scale=0.55, col=(255, 255, 255), th=1):
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), th + 3, cv2.LINE_AA)
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale, col, th, cv2.LINE_AA)


def defect_label(d) -> str:
    m = d.measures
    if d.type in ("hole", "pinhole", "dirt"):
        return f"{SHORT[d.type]} d={d.size_mm:.1f}mm"
    if d.type == "stain":
        return f"STAIN d={d.size_mm:.0f}mm"
    if d.type == "tear":
        return f"TEAR {d.size_mm:.1f}mm deep"
    if d.type == "missing_finger":
        return f"{m.get('finger', '').upper()} short {d.size_mm:.0f}mm"
    if d.type == "fused_fingers":
        return f"FUSED {m.get('fingers', '')}"
    if d.type == "cuff_defect":
        return f"CUFF {d.size_mm:.1f}mm"
    if d.type == "out_of_tolerance":
        return f"SIZE dL={m.get('length_dev_mm', 0):+.0f} dW={m.get('width_dev_mm', 0):+.0f}mm"
    return d.type


def draw_defects(img, defects, scale=1.0, offset=(0.0, 0.0)):
    for d in defects:
        col = DEFECT_COLOURS.get(d.type, (255, 255, 255))
        p = (int((d.x + offset[0]) * scale), int((d.y + offset[1]) * scale))
        r = int(max(10, d.size_mm * 1.6) * scale) if d.type in ("hole", "pinhole", "stain", "dirt") else int(22 * scale)
        if d.type == "out_of_tolerance":
            continue
        cv2.circle(img, p, r, col, 2, cv2.LINE_AA)
        _text(img, defect_label(d), (p[0] + r + 4, p[1] + 5), 0.5, col)


def draw_observation(img, obs, colour=(255, 255, 0), scale=1.0, show_defects=True):
    c = (obs.contour * scale).astype(np.int32)
    cv2.drawContours(img, [c], -1, colour, 2, cv2.LINE_AA)
    t = tuple(int(v * scale) for v in obs.tip_pt)
    b = tuple(int(v * scale) for v in obs.cuff_pt)
    cv2.arrowedLine(img, b, t, (255, 255, 255), 1, cv2.LINE_AA, tipLength=0.04)
    if obs.thumb_pt is not None:
        tp = tuple(int(v * scale) for v in obs.thumb_pt)
        cv2.circle(img, tp, 6, (255, 255, 255), 2, cv2.LINE_AA)
        _text(img, "thumb", (tp[0] + 8, tp[1]), 0.45)
    if show_defects:
        draw_defects(img, obs.defects, scale)


def draw_record(img, rec, scale=1.0, offset=(0.0, 0.0)):
    col = DECISION_COLOURS[rec.decision.label]
    if rec.contour is not None:
        cv2.drawContours(img, [(rec.contour * scale).astype(np.int32)], -1, col, 3, cv2.LINE_AA)
    draw_defects(img, rec.defects, scale, offset)
    x, y, w, h = cv2.boundingRect((rec.contour * scale).astype(np.int32)) if rec.contour is not None else (10, 10, 0, 0)
    lines = [f"#{rec.id} {rec.type} ({rec.type_conf:.2f})  hand {rec.hand}  {rec.direction.replace('_', ' ')}",
             f"L {rec.length_mm:.1f} mm  W {rec.width_mm:.1f} mm",
             f"{rec.decision.label}  q={rec.decision.score:.0f}"]
    y0 = max(20, y - 12 - 22 * (len(lines) - 1))
    for i, ln in enumerate(lines):
        _text(img, ln, (x, y0 + 22 * i), 0.6, col if i == 2 else (255, 255, 255), 2 if i == 2 else 1)


def render_frame(res, records_by_track: dict | None = None, scale=1.0):
    """Annotated copy of a FrameResult's rectified image."""
    img = res.rect.copy()
    if scale != 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    if res.learning:
        _text(img, "Learning empty plate - keep the plate clear", (20, 40), 0.9, (0, 255, 255), 2)
        return img
    records_by_track = records_by_track or {}
    for b in res.blobs:
        rec = records_by_track.get(b.track_id)
        if rec is not None:
            m = cv2.moments(b.contour)
            off = (0.0, 0.0)
            if rec.observation is not None and m["m00"] > 0:
                off = (m["m10"] / m["m00"] - rec.observation.center[0], m["m01"] / m["m00"] - rec.observation.center[1])
            rec.contour = b.contour
            draw_record(img, rec, scale, off)
        elif b.obs is not None:
            draw_observation(img, b.obs, (255, 255, 0), scale)
        else:
            cv2.drawContours(img, [(b.contour * scale).astype(np.int32)], -1, (160, 160, 160), 1, cv2.LINE_AA)
    return img
