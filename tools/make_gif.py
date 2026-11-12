"""Make a small README GIF from an annotated video.

  python tools/make_gif.py data/demo_annotated.mp4 media/demo.gif --start 40 --end 340 --step 3 --width 640
"""
import argparse

import cv2
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("video")
ap.add_argument("out")
ap.add_argument("--start", type=int, default=0)
ap.add_argument("--end", type=int, default=300)
ap.add_argument("--step", type=int, default=3)
ap.add_argument("--width", type=int, default=640)
ap.add_argument("--colors", type=int, default=64)
a = ap.parse_args()
cap = cv2.VideoCapture(a.video)
fps = cap.get(cv2.CAP_PROP_FPS) or 20
frames, i = [], 0
while True:
    ok, f = cap.read()
    if not ok or i >= a.end:
        break
    if i >= a.start and (i - a.start) % a.step == 0:
        h = int(f.shape[0] * a.width / f.shape[1])
        f = cv2.resize(f, (a.width, h), interpolation=cv2.INTER_AREA)
        frames.append(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)).quantize(colors=a.colors))
    i += 1
frames[0].save(a.out, save_all=True, append_images=frames[1:], duration=int(1000 * a.step / fps), loop=0, optimize=True)
import os
print(a.out, f"{os.path.getsize(a.out) / 1e6:.1f} MB, {len(frames)} frames")
