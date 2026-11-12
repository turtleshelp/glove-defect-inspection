"""Diagnostics for small holes: colour distance to the plate at each GT pinhole vs. the thresholds used.

  python tools/probe_pinholes.py data/test 30
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from gloveinspect import Inspector

data, n = Path(sys.argv[1]), int(sys.argv[2])
insp = Inspector()
insp.calibrate(cv2.imread(str(data / "calib.png")))
insp.learn_background([cv2.imread(str(data / "background.png"))])
labs = [json.loads(x) for x in (data / "labels.jsonl").read_text().splitlines() if x.strip()]
k = 0
for lab in labs:
    ph = [g for g in lab["defects"] if g["type"] == "pinhole"]
    if not ph:
        continue
    img = cv2.imread(str(data / lab["file"]))
    rect, valid, seg, blobs = insp._segment(img)
    for g in ph:
        x, y = insp.rect.img_to_rect([(g["x_px"], g["y_px"])])[0]
        x, y = int(round(x)), int(round(y))
        win = seg.de[y - 3:y + 4, x - 3:x + 4]
        ring = seg.de[y - 12:y + 13, x - 12:x + 13]
        print(f'{lab["id"]:4d} {lab["lighting"]:13s} d={g["diameter_mm"]:.2f}  de_min={win.min():5.1f}  '
              f'glove_de_median={np.median(ring):5.1f}  ratio={win.min() / np.median(ring):.2f}  in_mask={seg.mask[y, x] > 0}')
    k += 1
    if k >= n:
        break
