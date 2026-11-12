"""Headless command line.

  glove-inspect video  data/videos/demo.mp4 --calib data/videos/calib.png --out data/demo_annotated.mp4
  glove-inspect image  data/test/images/00012.png --calib data/test/calib.png --background data/test/background.png
  glove-inspect calibrate data/test/calib.png --save config/calibration.json
  glove-inspect train  --data data/train
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

from .config import load_config
from .inspector import Inspector
from .logger import CsvLogger
from .overlay import render_frame
from .sorter import SerialSorter, SimSorter


def _setup(a) -> Inspector:
    insp = Inspector(load_config(a.config))
    if a.calib:
        r = insp.calibrate(cv2.imread(str(a.calib)))
        print(f"calibrated: {r.px_per_mm} px/mm, checkerboard reprojection RMS {r.rms_mm:.3f} mm")
    return insp


def cmd_video(a):
    insp = _setup(a)
    cap = cv2.VideoCapture(int(a.source) if str(a.source).isdigit() else str(a.source))
    if not cap.isOpened():
        sys.exit(f"cannot open {a.source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 20
    cfg = insp.cfg
    logger = CsvLogger(cfg.path(cfg["logging"]["dir"]) if not a.log else a.log, str(a.source))
    sorter = SerialSorter(a.port, cfg["sorter"]["baud"]) if a.port else SimSorter()
    writer = None
    shown = {}
    n, t0 = 0, time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        res = insp.process(frame)
        for rec in res.finished:
            shown[rec.id] = rec
            logger.log(rec)
            sorter.sort(rec.id, rec.decision.label)
            print(f"glove #{rec.id}: {rec.type:8s} hand {rec.hand} {rec.direction:12s} L={rec.length_mm:6.1f} "
                  f"W={rec.width_mm:5.1f}  {','.join(d.type for d in rec.defects) or '-':40s} -> {rec.decision.label} "
                  f"(q={rec.decision.score})")
        if a.out:
            vis = render_frame(res, shown, a.scale)
            if writer is None:
                writer = cv2.VideoWriter(str(a.out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (vis.shape[1], vis.shape[0]))
            writer.write(vis)
        n += 1
    for rec in insp.flush():
        logger.log(rec)
        sorter.sort(rec.id, rec.decision.label)
    if writer:
        writer.release()
    print(f"{n} frames in {time.time() - t0:.1f} s ({n / max(1e-6, time.time() - t0):.1f} FPS); bins {sorter.bins}")
    print(f"log: {logger.gloves_path}")


def cmd_image(a):
    insp = _setup(a)
    if a.background:
        insp.learn_background([cv2.imread(str(a.background))])
    img = cv2.imread(str(a.image))
    if not insp.ready:
        sys.exit("need --background (photo of the empty plate)")
    res, recs = insp.inspect_still(img)
    for rec in recs:
        print(f"{rec.type} (p={rec.type_conf}) hand {rec.hand} {rec.direction} L={rec.length_mm} W={rec.width_mm} "
              f"-> {rec.decision}")
        for d in rec.defects:
            print(f"   {d.type:16s} {d.size_mm:6.2f} mm  {d.measures}")
    if a.out:
        cv2.imwrite(str(a.out), render_frame(res, {r.id: r for r in recs}))


def cmd_calibrate(a):
    insp = _setup(argparse.Namespace(config=a.config, calib=a.image))
    if a.save:
        insp.rect.save(a.save)
        print("saved", a.save)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="glove-inspect", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("video", help="inspect a video file or camera index")
    v.add_argument("source")
    v.add_argument("--calib", type=Path)
    v.add_argument("--out", type=Path)
    v.add_argument("--scale", type=float, default=0.6)
    v.add_argument("--log", type=Path)
    v.add_argument("--port", help="Arduino serial port, e.g. COM5")
    i = sub.add_parser("image", help="inspect one image")
    i.add_argument("image", type=Path)
    i.add_argument("--calib", type=Path)
    i.add_argument("--background", type=Path)
    i.add_argument("--out", type=Path)
    c = sub.add_parser("calibrate")
    c.add_argument("image", type=Path)
    c.add_argument("--save", type=Path)
    t = sub.add_parser("train")
    t.add_argument("--data", default="data/train")
    a = ap.parse_args(argv)
    if a.cmd == "video":
        cmd_video(a)
    elif a.cmd == "image":
        cmd_image(a)
    elif a.cmd == "calibrate":
        cmd_calibrate(a)
    else:
        from .train import main as train_main
        train_main(["--data", a.data] + (["--config", a.config] if a.config else []))


if __name__ == "__main__":
    main()
