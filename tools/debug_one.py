"""Debug one still: print detections next to ground truth (in rectified px) and save a zoomed view.

  python tools/debug_one.py data/sample 12 data/debug.png
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from gloveinspect import Inspector
from gloveinspect.overlay import draw_record

data, idx, out = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
insp = Inspector()
insp.calibrate(cv2.imread(str(data / "calib.png")))
insp.learn_background([cv2.imread(str(data / "background.png"))])
lab = [json.loads(x) for x in (data / "labels.jsonl").read_text().splitlines() if x.strip()][idx]
img = cv2.imread(str(data / lab["file"]))
res, recs = insp.inspect_still(img)
vis = res.rect.copy()
for g in lab["defects"]:
    p = insp.rect.img_to_rect([(g["x_px"], g["y_px"])])[0]
    print("GT ", g["type"], np.round(p, 1), {k: v for k, v in g.items() if k not in ("type", "x_px", "y_px")})
    cv2.drawMarker(vis, tuple(int(v) for v in p), (0, 255, 255), cv2.MARKER_TILTED_CROSS, 30, 2)
for r in recs:
    print("glove", r.type, r.hand, r.direction, r.length_mm, r.width_mm, r.decision)
    print("stats", r.observation.stats)
    for d in r.defects:
        print("DET", d.type, (round(d.x, 1), round(d.y, 1)), round(d.size_mm, 2), d.measures)
    draw_record(vis, r)
    x, y, w, h = cv2.boundingRect(r.contour)
    vis = vis[max(0, y - 60):y + h + 60, max(0, x - 60):x + w + 60]
cv2.imwrite(out, vis)
