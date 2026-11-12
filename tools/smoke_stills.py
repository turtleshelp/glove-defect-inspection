"""Quick visual check: inspect the first N stills of a dataset and write an annotated contact sheet.

  python tools/smoke_stills.py data/sample data/smoke.jpg --n 15
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from gloveinspect import Inspector
from gloveinspect.overlay import draw_record

ap = argparse.ArgumentParser()
ap.add_argument("dataset", type=Path)
ap.add_argument("out", type=Path)
ap.add_argument("--n", type=int, default=15)
ap.add_argument("--start", type=int, default=0)
a = ap.parse_args()

insp = Inspector()
r = insp.calibrate(cv2.imread(str(a.dataset / "calib.png")))
print(f"calibration rms {r.rms_mm:.3f} mm, rectified size {r.size}")
insp.learn_background([cv2.imread(str(a.dataset / "background.png"))])
labs = [json.loads(line) for line in (a.dataset / "labels.jsonl").read_text().splitlines() if line.strip()]
tiles = []
for lab in labs[a.start:a.start + a.n]:
    img = cv2.imread(str(a.dataset / lab["file"]))
    t0 = time.perf_counter()
    res, recs = insp.inspect_still(img)
    dt = (time.perf_counter() - t0) * 1000
    vis = res.rect.copy()
    for rec in recs:
        draw_record(vis, rec)
    gt = sorted(d["type"] for d in lab["defects"])
    det = sorted(d.type for rec in recs for d in rec.defects)
    line = (f'{lab["id"]:4d} GT {lab["type"]:8s} {lab["hand"]} {lab["direction"]:12s} L={lab["length_mm"]:.1f} '
            f'W={lab["width_mm"]:.1f} {gt}')
    print(line)
    for rec in recs:
        print(f'      -> {rec.type:8s} {rec.hand} {rec.direction:12s} L={rec.length_mm:.1f} W={rec.width_mm:.1f} '
              f'{det} {rec.decision.label} ({dt:.0f} ms)')
    if not recs:
        print("      -> no glove found")
    cv2.putText(vis, f'GT: {lab["type"]} {lab["hand"]} {gt}', (20, vis.shape[0] - 30), 0, 1.0, (0, 255, 255), 2)
    tiles.append(cv2.resize(vis, (640, 360)))
while len(tiles) % 3:
    tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite(str(a.out), np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]))
