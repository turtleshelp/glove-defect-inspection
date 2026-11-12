"""Run the full video pipeline (tracking + voting + fuzzy decision) on a synthetic conveyor video and
compare each finished glove with the ground truth, in order of appearance.

  python results/evaluate_video.py --video data/videos/demo.mp4 --out results --gif media/demo.gif
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from gloveinspect import Inspector
from gloveinspect.decision import rule_table_decision
from gloveinspect.overlay import render_frame


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, default=Path("data/videos/demo.mp4"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--gif", type=Path, default=None)
    ap.add_argument("--gif-frames", type=str, default="0:99999")
    ap.add_argument("--mp4", type=Path, default=None)
    a = ap.parse_args(argv)
    gt = [json.loads(x) for x in a.video.with_suffix(".jsonl").read_text().splitlines() if x.strip()]
    insp = Inspector()
    insp.calibrate(cv2.imread(str(a.video.with_name("calib.png"))))
    cap = cv2.VideoCapture(str(a.video))
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 20
    recs, shown, gif_frames = [], {}, []
    g0, g1 = (int(v) for v in a.gif_frames.split(":"))
    writer = None
    n, t_proc = 0, 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.perf_counter()
        res = insp.process(frame)
        t_proc += time.perf_counter() - t0
        for r in res.finished:
            recs.append(r)
            shown[r.id] = r
        if a.gif or a.mp4:
            vis = render_frame(res, shown, 0.5)
            if a.mp4:
                if writer is None:
                    writer = cv2.VideoWriter(str(a.mp4), cv2.VideoWriter_fourcc(*"mp4v"), fps_src, (vis.shape[1], vis.shape[0]))
                writer.write(vis)
            if a.gif and g0 <= n < g1 and n % 2 == 0:
                gif_frames.append(cv2.resize(vis, None, fx=0.8, fy=0.8, interpolation=cv2.INTER_AREA))
        n += 1
    recs += insp.flush()
    if writer:
        writer.release()

    rows = []
    for i, g in enumerate(gt):
        r = recs[i] if i < len(recs) else None
        gdec = rule_table_decision(g["defects"])
        if r is None:
            rows.append(dict(id=i, gt=g, found=False))
            continue
        det = sorted({d.type for d in r.defects})
        rows.append(dict(id=i, found=True, type_ok=r.type == g["type"], hand_ok=r.hand == g["hand"],
                         dir_ok=r.direction == g["direction"], defects_ok=det == sorted(g["defects"]),
                         dec_ok=r.decision.label == gdec, gt_type=g["type"], type=r.type, gt_defects=g["defects"],
                         defects=det, gt_decision=gdec, decision=r.decision.label, frames=r.n_obs,
                         length_err=round(r.length_mm - g["length_mm"], 2), width_err=round(r.width_mm - g["width_mm"], 2)))
    found = [r for r in rows if r["found"]]
    summary = dict(video=str(a.video), frames=n, gloves_gt=len(gt), gloves_counted=len(recs),
                   processing_fps=round(n / t_proc, 1),
                   type_correct=sum(r["type_ok"] for r in found), hand_correct=sum(r["hand_ok"] for r in found),
                   direction_correct=sum(r["dir_ok"] for r in found),
                   defect_set_exact=sum(r["defects_ok"] for r in found),
                   decision_correct=sum(r["dec_ok"] for r in found),
                   mean_frames_per_glove=round(float(np.mean([r["frames"] for r in found])), 1) if found else 0,
                   gloves=rows)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "video_eval.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "gloves"}, indent=1))
    for r in rows:
        if r["found"]:
            print(f"{r['id']:2d} {r['gt_type']:8s}->{r['type']:8s} GT {r['gt_defects']} DET {r['defects']} "
                  f"{r['gt_decision']}->{r['decision']}  dL={r['length_err']:+.1f}")
    if a.gif and gif_frames:
        from PIL import Image
        imgs = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)).quantize(colors=96, method=Image.Quantize.MEDIANCUT)
                for f in gif_frames]
        a.gif.parent.mkdir(parents=True, exist_ok=True)
        imgs[0].save(a.gif, save_all=True, append_images=imgs[1:], duration=int(2000 / fps_src), loop=0, optimize=True)
        print("gif:", a.gif, f"{a.gif.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
