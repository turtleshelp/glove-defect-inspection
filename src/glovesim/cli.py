"""Command-line generator for synthetic datasets.

  glovesim stills --out data/test --n 900 --seed 2      # one glove per image + labels.jsonl
  glovesim video  --out data/videos --name demo --n-gloves 12 --seed 7

Every dataset folder also gets calib.png (checkerboard), background.png (empty plate) and meta.json.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

from .model import DEFECT_TYPES, GLOVE_TYPES, TOLERANCE_MM, nominal_dimensions, render_glove, sample_glove
from .scene import BOARD, CAM_H, CAM_W, LIGHTING, WORLD_H, WORLD_PPM, WORLD_W, Scene

def _np(o):
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(type(o))


_dumps = json.dumps


def _jdumps(obj, **kw):
    return _dumps(obj, default=_np, **kw)


json.dumps = _jdumps  # labels contain NumPy scalars


def _nominals():
    return {t: nominal_dimensions(t) for t in GLOVE_TYPES}


def choose_defects(rng, p_clean=0.35):
    if rng.random() < p_clean:
        return []
    k = 1 if rng.random() < 0.7 else 2
    return list(rng.choice(DEFECT_TYPES, size=k, replace=False))


def write_common(out: Path, scene: Scene, rng):
    out.mkdir(parents=True, exist_ok=True)
    nominal = LIGHTING["nominal"]
    cv2.imwrite(str(out / "calib.png"), scene.finish(scene.to_camera(scene.calibration_world()), nominal, rng))
    cv2.imwrite(str(out / "background.png"), scene.finish(scene.to_camera(scene.plate.copy()), nominal, rng))
    meta = dict(world_px_per_mm=WORLD_PPM, camera=[CAM_W, CAM_H], board=BOARD, tolerance_mm=TOLERANCE_MM,
                nominal_mm=_nominals(), H_world_to_cam=scene.H.tolist(), plate_bgr=list(scene.plate_bgr))
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    return meta


def build_label(scene: Scene, glove: dict, M, spec, meta, angle, lighting, colour_name, idx, file):
    gt = []
    for d in glove["gt"]:
        if d["type"] == "out_of_tolerance":
            continue
        (cx, cy), = scene.local_to_cam(glove, M, [(d["x"], d["y"])])
        e = {k: v for k, v in d.items() if k not in ("x", "y")}
        e.update(x_px=round(float(cx), 1), y_px=round(float(cy), 1))
        gt.append(e)
    nom = meta["nominal_mm"][spec.type]
    tol = meta["tolerance_mm"]
    out_tol = abs(glove["length_mm"] - nom["length"]) > tol["length"] or abs(glove["width_mm"] - nom["width"]) > tol["width"]
    if out_tol:
        (cx, cy), = scene.local_to_cam(glove, M, [(0.0, glove["length_mm"] / 2)])
        gt.append(dict(type="out_of_tolerance", x_px=round(float(cx), 1), y_px=round(float(cy), 1),
                       length_dev_mm=round(glove["length_mm"] - nom["length"], 2),
                       width_dev_mm=round(glove["width_mm"] - nom["width"], 2)))
    a = (angle % 360 + 360) % 360
    return dict(id=idx, file=file, type=spec.type, hand=spec.hand,
                direction="fingers_up" if math.cos(math.radians(a)) > 0 else "fingers_down",
                angle_deg=round(a, 2), lighting=lighting, colour=colour_name,
                length_mm=round(glove["length_mm"], 2), width_mm=round(glove["width_mm"], 2),
                scale=round(spec.scale, 4), defects=gt)


def _random_angle(rng):
    a = rng.uniform(-50, 50)
    return a if rng.random() < 0.5 else 180 + a


def _fits(scene: Scene, a_w, margin=25):
    ys, xs = np.nonzero(a_w > 0.3)
    if len(xs) == 0:
        return False
    pts = np.array([[xs.min(), ys.min()], [xs.max(), ys.min()], [xs.max(), ys.max()], [xs.min(), ys.max()]], np.float32)
    c = scene.world_to_cam(pts)
    return c[:, 0].min() > margin and c[:, 1].min() > margin and c[:, 0].max() < CAM_W - margin and c[:, 1].max() < CAM_H - margin


def make_stills(out: Path, n: int, seed: int, p_clean: float = 0.35):
    rng = np.random.default_rng(seed)
    scene = Scene(rng)
    meta = write_common(out, scene, rng)
    (out / "images").mkdir(exist_ok=True)
    types = list(GLOVE_TYPES)
    conds = list(LIGHTING)
    lines = []
    for i in range(n):
        gtype = types[i % 3]
        hand = "R" if rng.random() < 0.5 else "L"
        defects = choose_defects(rng, p_clean)
        cname, colour = scene.sample_glove_colour()
        spec = sample_glove(rng, gtype, hand, colour, defects)
        glove = render_glove(spec, rng, plate_bgr=scene.plate_bgr)
        lighting = conds[(i // 3) % len(conds)]
        for _ in range(50):
            angle = _random_angle(rng)
            center = (WORLD_W / 2 + rng.uniform(-60, 60), WORLD_H / 2 + rng.uniform(-25, 25))
            M, _ = scene.glove_affine(glove, center, angle)
            world, a_w = scene.composite(scene.plate.copy(), glove, M)
            if _fits(scene, a_w):
                break
        img = scene.finish(scene.to_camera(world), scene.jitter(LIGHTING[lighting]))
        fname = f"images/{i:05d}.png"
        cv2.imwrite(str(out / fname), img)
        lines.append(json.dumps(build_label(scene, glove, M, spec, meta, angle, lighting, cname, i, fname)))
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{n}")
    (out / "labels.jsonl").write_text("\n".join(lines) + "\n")
    print(f"wrote {n} images to {out}")


def make_video(out: Path, name: str, n_gloves: int, seed: int, fps: int = 20, speed_mm_s: float = 180.0,
               p_clean: float = 0.35):
    rng = np.random.default_rng(seed)
    scene = Scene(rng)
    meta = write_common(out, scene, rng)
    types = list(GLOVE_TYPES)
    gloves = []
    for i in range(n_gloves):
        gtype = types[int(rng.integers(3))]
        hand = "R" if rng.random() < 0.5 else "L"
        defects = choose_defects(rng, p_clean)
        cname, colour = scene.sample_glove_colour()
        spec = sample_glove(rng, gtype, hand, colour, defects)
        glove = render_glove(spec, rng, plate_bgr=scene.plate_bgr)
        for _ in range(100):
            angle = _random_angle(rng)
            yc = WORLD_H / 2 + rng.uniform(-20, 20)
            M, _ = scene.glove_affine(glove, (WORLD_W / 2, yc), angle)
            _, a_w = scene.composite(scene.plate.copy(), glove, M)
            if _fits(scene, a_w, 40):
                break
        gloves.append((spec, glove, angle, yc, cname))
    pitch_mm = 520.0
    start_mm = -260.0  # centre of the first glove when the gloves start moving (off-screen left)
    empty_s = 1.5
    total_mm = (n_gloves - 1) * pitch_mm + (WORLD_W / WORLD_PPM) + 520
    n_frames = int(empty_s * fps + total_mm / speed_mm_s * fps)
    writer = cv2.VideoWriter(str(out / f"{name}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (CAM_W, CAM_H))
    labels = []
    for k, (spec, glove, angle, yc, cname) in enumerate(gloves):
        M, _ = scene.glove_affine(glove, (WORLD_W / 2, yc), angle)
        lab = build_label(scene, glove, M, spec, meta, angle, "video", cname, k, f"{name}.mp4")
        lab["defects"] = sorted({d["type"] for d in lab["defects"]})  # positions change while moving
        labels.append(lab)
    for f in range(n_frames):
        t = f / fps
        world = scene.plate.copy()
        moving = max(0.0, t - empty_s)
        for k, (spec, glove, angle, yc, cname) in enumerate(gloves):
            x_mm = start_mm + moving * speed_mm_s - k * pitch_mm
            if x_mm < -300 or x_mm > WORLD_W / WORLD_PPM + 300:
                continue
            M, _ = scene.glove_affine(glove, (x_mm * WORLD_PPM, yc), angle)
            world, _ = scene.composite(world, glove, M)
        cond = dict(LIGHTING["nominal"])
        cond["gain"] = 1.0 + 0.22 * math.sin(2 * math.pi * t / 16.0)
        cond["gx"] = 0.25 * math.sin(2 * math.pi * t / 23.0)
        cond["noise"] = 3.0
        writer.write(scene.finish(scene.to_camera(world), cond))
    writer.release()
    (out / f"{name}.jsonl").write_text("\n".join(json.dumps(x) for x in labels) + "\n")
    print(f"wrote {out / (name + '.mp4')}: {n_frames} frames, {n_gloves} gloves")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="glovesim", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stills")
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--n", type=int, default=900)
    s.add_argument("--seed", type=int, default=1)
    s.add_argument("--p-clean", type=float, default=0.35)
    v = sub.add_parser("video")
    v.add_argument("--out", type=Path, required=True)
    v.add_argument("--name", default="demo")
    v.add_argument("--n-gloves", type=int, default=12)
    v.add_argument("--seed", type=int, default=7)
    v.add_argument("--fps", type=int, default=20)
    sub.add_parser("nominal")
    a = ap.parse_args(argv)
    if a.cmd == "stills":
        make_stills(a.out, a.n, a.seed, a.p_clean)
    elif a.cmd == "video":
        make_video(a.out, a.name, a.n_gloves, a.seed, a.fps)
    else:
        print(json.dumps(_nominals(), indent=1))


if __name__ == "__main__":
    main()
