"""Train the glove-type MLP and learn golden shape statistics from a training dataset.

  python -m gloveinspect.train --data data/train

Only the TRAINING set (different random seed from the test set) is used here.
The golden statistics (expected finger lengths, finger positions, thumb protrusion, finger valley
height) are the medians over defect-free training gloves of each type - the synthetic equivalent of
measuring a few known-good reference gloves.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .classify import FEATURES, TypeClassifier, feature_vector, learn_golden
from .config import load_config
from .inspector import Inspector


def collect(inspector: Inspector, data: Path, limit: int | None = None):
    labs = [json.loads(x) for x in (data / "labels.jsonl").read_text().splitlines() if x.strip()]
    if limit:
        labs = labs[:limit]
    rows = []
    for i, lab in enumerate(labs):
        img = cv2.imread(str(data / lab["file"]))
        rect, valid, seg, blobs = inspector._segment(img)
        blobs = [b for b in blobs if b.complete]
        if len(blobs) != 1:
            continue
        obs = inspector._analyse(seg, blobs[0].contour)
        if obs is None:
            continue
        rows.append((lab, obs))
        if (i + 1) % 100 == 0:
            print(f"  analysed {i + 1}/{len(labs)}")
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/train"))
    ap.add_argument("--config", default=None)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    insp = Inspector(cfg)
    insp.classifier = TypeClassifier(None, cfg["glove_types"])  # untrained: features do not depend on it
    insp.golden = {}
    insp.calibrate(cv2.imread(str(a.data / "calib.png")))
    insp.learn_background([cv2.imread(str(a.data / "background.png"))])
    rows = collect(insp, a.data, a.limit)
    print(f"{len(rows)} gloves analysed")

    X = np.stack([feature_vector(o.features) for _, o in rows])
    y = [lab["type"] for lab, _ in rows]
    clf = TypeClassifier.train(X, y)
    clf.save(cfg.path(cfg["models"]["type_classifier"]))
    train_acc = float(np.mean(clf.model.predict(X) == np.array(y)))

    stats = {}
    for lab, o in rows:
        if not lab["defects"]:
            stats.setdefault(lab["type"], []).append(o.stats)
    golden = learn_golden(stats)
    gp = cfg.path(cfg["models"]["golden"])
    gp.parent.mkdir(parents=True, exist_ok=True)
    gp.write_text(json.dumps(golden, indent=1))
    report = dict(n_train=len(rows), features=FEATURES, train_accuracy=round(train_acc, 4),
                  golden_n={k: v["n"] for k, v in golden.items()})
    (gp.parent / "training_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    print(json.dumps(golden, indent=1))


if __name__ == "__main__":
    main()
