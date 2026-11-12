"""Contact sheet of a generated stills dataset with its ground truth drawn on top.

  python tools/preview_dataset.py data/test media/dataset_preview.jpg --n 12
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("dataset", type=Path)
ap.add_argument("out", type=Path)
ap.add_argument("--n", type=int, default=12)
ap.add_argument("--start", type=int, default=0)
ap.add_argument("--cols", type=int, default=3)
a = ap.parse_args()

labs = [json.loads(line) for line in (a.dataset / "labels.jsonl").read_text().splitlines() if line.strip()]
tiles = []
for lab in labs[a.start:a.start + a.n]:
    im = cv2.imread(str(a.dataset / lab["file"]))
    for g in lab["defects"]:
        c = (int(g["x_px"]), int(g["y_px"]))
        cv2.circle(im, c, 30, (0, 0, 255), 3)
        cv2.putText(im, g["type"], (c[0] + 34, c[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
    txt = f'GT: {lab["type"]} {lab["hand"]} {lab["direction"]} | {lab["lighting"]}'
    cv2.putText(im, txt, (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 8)
    cv2.putText(im, txt, (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
    tiles.append(cv2.resize(im, (640, 360), interpolation=cv2.INTER_AREA))
while len(tiles) % a.cols:
    tiles.append(np.zeros_like(tiles[0]))
rows = [np.hstack(tiles[i:i + a.cols]) for i in range(0, len(tiles), a.cols)]
a.out.parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(a.out), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
print("wrote", a.out)
