"""Evaluate the inspector on a labelled synthetic test set (never used for training).

  python results/evaluate.py --data data/test --out results

Writes metrics.json, metrics.md, failures.csv and PNG charts (confusion matrices, per-defect
precision/recall, measurement errors, results per lighting condition).
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from gloveinspect import Inspector
from gloveinspect.decision import rule_table_decision

DEFECTS = ["hole", "pinhole", "tear", "missing_finger", "fused_fingers", "stain", "dirt", "cuff_defect", "out_of_tolerance"]
RADIUS_MM = {"hole": 12, "pinhole": 12, "stain": 15, "dirt": 12, "tear": 15, "missing_finger": 45, "fused_fingers": 40,
             "cuff_defect": 40}
GT_SIZE = {"hole": "diameter_mm", "pinhole": "diameter_mm", "stain": "diameter_mm", "dirt": "diameter_mm",
           "tear": "depth_mm", "cuff_defect": "max_dev_mm", "missing_finger": "missing_length_mm"}
TYPES = ["exam", "longcuff", "garden"]
DECISIONS = ["PASS", "REWORK", "REJECT"]

# chart styling (reference palette of the dataviz method)
BLUE_RAMP = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"


def match_defects(gt, det, ppm):
    """Greedy nearest matching per defect type. Returns list of (gt, det|None), unmatched det list."""
    pairs, used = [], set()
    for t in DEFECTS:
        if t == "out_of_tolerance":
            continue
        g_t = [g for g in gt if g["type"] == t]
        d_t = [(i, d) for i, d in enumerate(det) if d.type == t]
        cand = []
        for gi, g in enumerate(g_t):
            for i, d in d_t:
                dist = np.hypot(g["rx"] - d.x, g["ry"] - d.y) / ppm
                if dist <= RADIUS_MM[t]:
                    cand.append((dist, gi, i))
        cand.sort()
        g_done = set()
        for dist, gi, i in cand:
            if gi in g_done or i in used:
                continue
            g_done.add(gi)
            used.add(i)
            pairs.append((g_t[gi], det[i]))
        for gi, g in enumerate(g_t):
            if gi not in g_done:
                pairs.append((g, None))
    unmatched = [d for i, d in enumerate(det) if i not in used and d.type != "out_of_tolerance"]
    return pairs, unmatched


def cross_type(gt_missed, det_unmatched, ppm):
    """For the confusion matrix: was a missed GT defect detected as another type at the same place?"""
    out = []
    used = set()
    for g in gt_missed:
        best = None
        for i, d in enumerate(det_unmatched):
            if i in used:
                continue
            dist = np.hypot(g["rx"] - d.x, g["ry"] - d.y) / ppm
            if dist <= RADIUS_MM[g["type"]] and (best is None or dist < best[0]):
                best = (dist, i)
        if best:
            used.add(best[1])
            out.append((g["type"], det_unmatched[best[1]].type))
        else:
            out.append((g["type"], "missed"))
    fa = [d.type for i, d in enumerate(det_unmatched) if i not in used]
    return out, fa


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    f = 2 * p * r / (p + r) if p + r and not np.isnan(p + r) else float("nan")
    return p, r, f


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/test"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)

    insp = Inspector()
    if not insp.classifier.trained:
        raise SystemExit("train first: python -m gloveinspect.train --data data/train")
    r = insp.calibrate(cv2.imread(str(a.data / "calib.png")))
    insp.learn_background([cv2.imread(str(a.data / "background.png"))])
    ppm = r.px_per_mm
    labs = [json.loads(x) for x in (a.data / "labels.jsonl").read_text().splitlines() if x.strip()]
    if a.limit:
        labs = labs[:a.limit]

    rows, times = [], []
    tp, fp, fn = Counter(), Counter(), Counter()
    meas_err = defaultdict(list)
    confusion_def = Counter()
    false_alarms = Counter()
    per_light = defaultdict(lambda: Counter())
    failures = []
    for k, lab in enumerate(labs):
        img = cv2.imread(str(a.data / lab["file"]))
        t0 = time.perf_counter()
        res, recs = insp.inspect_still(img)
        times.append((time.perf_counter() - t0) * 1000)
        gt = [dict(g) for g in lab["defects"]]
        for g in gt:
            g["rx"], g["ry"] = insp.rect.img_to_rect([(g["x_px"], g["y_px"])])[0]
        gt_types = sorted({g["type"] for g in gt})
        gt_dec = rule_table_decision(gt_types)
        L = lab["lighting"]
        if not recs:
            rows.append(dict(id=lab["id"], found=False, lighting=L, gt_type=lab["type"], gt_dec=gt_dec))
            for g in gt:
                fn[g["type"]] += 1
                confusion_def[(g["type"], "missed")] += 1
            failures.append([lab["id"], lab["file"], "glove not found", "", ""])
            per_light[L]["n"] += 1
            continue
        rec = max(recs, key=lambda q: cv2.contourArea(q.contour))
        det = list(rec.defects)
        pairs, unmatched = match_defects(gt, det, ppm)
        missed = []
        for g, d in pairs:
            if d is None:
                fn[g["type"]] += 1
                missed.append(g)
            else:
                tp[g["type"]] += 1
                confusion_def[(g["type"], g["type"])] += 1
                key = GT_SIZE.get(g["type"])
                if key and key in g:
                    meas_err[g["type"]].append(d.size_mm - g[key])
        cross, fa = cross_type(missed, unmatched, ppm)
        for gt_t, det_t in cross:
            confusion_def[(gt_t, det_t)] += 1
        for d in unmatched:
            fp[d.type] += 1
        for t in fa:
            false_alarms[t] += 1
        gt_oot = "out_of_tolerance" in gt_types
        det_oot = any(d.type == "out_of_tolerance" for d in det)
        if gt_oot and det_oot:
            tp["out_of_tolerance"] += 1
            confusion_def[("out_of_tolerance", "out_of_tolerance")] += 1
        elif gt_oot:
            fn["out_of_tolerance"] += 1
            confusion_def[("out_of_tolerance", "missed")] += 1
        elif det_oot:
            fp["out_of_tolerance"] += 1
            false_alarms["out_of_tolerance"] += 1
        row = dict(id=lab["id"], found=True, lighting=L, gt_type=lab["type"], type=rec.type,
                   gt_hand=lab["hand"], hand=rec.hand, gt_dir=lab["direction"], dir=rec.direction,
                   dl=rec.length_mm - lab["length_mm"], dw=rec.width_mm - lab["width_mm"], clean=not gt,
                   gt_dec=gt_dec, dec=rec.decision.label, score=rec.decision.score,
                   gt_any=bool(gt), det_any=bool(det))
        rows.append(row)
        n_tp = sum(1 for g, d in pairs if d is not None) + int(gt_oot and det_oot)
        n_gt = len(gt) + 0
        per_light[L]["n"] += 1
        per_light[L]["type_ok"] += row["type"] == row["gt_type"]
        per_light[L]["hand_ok"] += row["hand"] == row["gt_hand"]
        per_light[L]["dec_ok"] += row["dec"] == row["gt_dec"]
        per_light[L]["tp"] += n_tp
        per_light[L]["gt"] += len([g for g in gt]) + 0
        per_light[L]["det"] += len([d for d in det])
        per_light[L]["abs_dl"] += abs(row["dl"])
        bad = []
        if row["type"] != row["gt_type"]:
            bad.append(f"type {row['gt_type']}->{row['type']}")
        if row["hand"] != row["gt_hand"]:
            bad.append(f"hand {row['gt_hand']}->{row['hand']}")
        if row["dir"] != row["gt_dir"]:
            bad.append("direction")
        if missed:
            bad.append("missed " + ",".join(g["type"] for g in missed))
        if unmatched:
            bad.append("false " + ",".join(d.type for d in unmatched))
        if row["dec"] != gt_dec:
            bad.append(f"decision {gt_dec}->{row['dec']}")
        if bad:
            failures.append([lab["id"], lab["file"], "; ".join(bad), ",".join(gt_types),
                             ",".join(sorted(d.type for d in det))])
        if (k + 1) % 100 == 0:
            print(f"  {k + 1}/{len(labs)}")

    found = [r_ for r_ in rows if r_["found"]]
    n = len(rows)
    acc = lambda key, gkey: float(np.mean([r_[key] == r_[gkey] for r_ in found]))  # noqa: E731
    metrics = dict(
        dataset=str(a.data), n_gloves=n, n_found=len(found),
        type_accuracy=acc("type", "gt_type"),
        hand_accuracy=acc("hand", "gt_hand"),
        hand_unknown_rate=float(np.mean([r_["hand"] == "?" for r_ in found])),
        direction_accuracy=acc("dir", "gt_dir"),
        decision_accuracy=float(np.mean([r_["dec"] == r_["gt_dec"] for r_ in found])),
        defective_glove_detection=dict(
            recall=float(np.mean([r_["det_any"] for r_ in found if r_["gt_any"]])),
            false_alarm_rate=float(np.mean([r_["det_any"] for r_ in found if not r_["gt_any"]])),
        ),
        length_error_mm=dict(mae=float(np.mean([abs(r_["dl"]) for r_ in found])),
                             p95=float(np.percentile([abs(r_["dl"]) for r_ in found], 95)),
                             bias=float(np.mean([r_["dl"] for r_ in found]))),
        width_error_mm=dict(mae=float(np.mean([abs(r_["dw"]) for r_ in found])),
                            p95=float(np.percentile([abs(r_["dw"]) for r_ in found], 95)),
                            bias=float(np.mean([r_["dw"] for r_ in found]))),
        time_ms_per_image=dict(mean=float(np.mean(times)), p95=float(np.percentile(times, 95))),
        defects={},
        measurement_error_mm={},
        per_lighting={},
    )
    for t in DEFECTS:
        p, r_, f = prf(tp[t], fp[t], fn[t])
        metrics["defects"][t] = dict(n_gt=tp[t] + fn[t], tp=tp[t], fp=fp[t], fn=fn[t], precision=p, recall=r_, f1=f)
    for t, errs in meas_err.items():
        e = np.array(errs)
        metrics["measurement_error_mm"][t] = dict(n=len(e), mae=float(np.mean(np.abs(e))), bias=float(np.mean(e)),
                                                  p95=float(np.percentile(np.abs(e), 95)))
    for L, c in per_light.items():
        p = c["tp"] / c["det"] if c["det"] else float("nan")
        r_ = c["tp"] / c["gt"] if c["gt"] else float("nan")
        metrics["per_lighting"][L] = dict(n=c["n"], type_accuracy=c["type_ok"] / c["n"], hand_accuracy=c["hand_ok"] / c["n"],
                                          decision_accuracy=c["dec_ok"] / c["n"], defect_precision=p, defect_recall=r_,
                                          length_mae_mm=c["abs_dl"] / c["n"])
    tc = Counter((r_["gt_type"], r_["type"]) for r_ in found)
    dcm = Counter((r_["gt_dec"], r_["dec"]) for r_ in found)
    metrics["type_confusion"] = {f"{g}->{p}": v for (g, p), v in tc.items()}
    metrics["decision_confusion"] = {f"{g}->{p}": v for (g, p), v in dcm.items()}
    metrics["defect_confusion"] = {f"{g}->{p}": v for (g, p), v in confusion_def.items()}
    metrics["false_alarms"] = dict(false_alarms)

    (a.out / "metrics.json").write_text(json.dumps(metrics, indent=1, default=float))
    with open(a.out / "failures.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "file", "problems", "gt_defects", "detected_defects"])
        w.writerows(failures)
    write_markdown(metrics, a.out / "metrics.md")
    plots(metrics, tc, dcm, confusion_def, false_alarms, meas_err, a.out)
    print(open(a.out / "metrics.md", encoding="utf-8").read())


def fmt(x, pct=True):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    return f"{100 * x:.1f} %" if pct else f"{x:.2f}"


def write_markdown(m, path):
    L = []
    L.append(f"# Evaluation – synthetic test set ({m['n_gloves']} gloves)\n")
    L.append("All numbers are on **synthetic** images (generated with a different seed than the training set). "
             "They are not real-world results.\n")
    L.append("| Glove-level metric | Value |\n|---|---|")
    L.append(f"| Gloves found | {m['n_found']} / {m['n_gloves']} |")
    L.append(f"| Glove type accuracy (3 classes) | {fmt(m['type_accuracy'])} |")
    L.append(f"| Left/right hand accuracy | {fmt(m['hand_accuracy'])} (undecided: {fmt(m['hand_unknown_rate'])}) |")
    L.append(f"| Top/bottom (finger direction) accuracy | {fmt(m['direction_accuracy'])} |")
    L.append(f"| Length error | MAE {m['length_error_mm']['mae']:.2f} mm, 95th pct {m['length_error_mm']['p95']:.2f} mm |")
    L.append(f"| Width error | MAE {m['width_error_mm']['mae']:.2f} mm, 95th pct {m['width_error_mm']['p95']:.2f} mm |")
    dd = m["defective_glove_detection"]
    L.append(f"| Defective gloves flagged (any defect) | {fmt(dd['recall'])} |")
    L.append(f"| Good gloves wrongly flagged | {fmt(dd['false_alarm_rate'])} |")
    L.append(f"| PASS/REWORK/REJECT decision accuracy | {fmt(m['decision_accuracy'])} |")
    L.append(f"| Processing time per image (CPU) | {m['time_ms_per_image']['mean']:.0f} ms mean |\n")
    L.append("| Defect | GT count | Precision | Recall | F1 | Size error (MAE) |\n|---|---|---|---|---|---|")
    for t, d in m["defects"].items():
        me = m["measurement_error_mm"].get(t)
        mes = f"{me['mae']:.2f} mm (bias {me['bias']:+.2f})" if me else "–"
        L.append(f"| {t} | {d['n_gt']} | {fmt(d['precision'])} | {fmt(d['recall'])} | {fmt(d['f1'])} | {mes} |")
    L.append("\n| Lighting | n | Type acc. | Hand acc. | Defect precision | Defect recall | Decision acc. | Length MAE |")
    L.append("|---|---|---|---|---|---|---|---|")
    for k, d in m["per_lighting"].items():
        L.append(f"| {k} | {d['n']} | {fmt(d['type_accuracy'])} | {fmt(d['hand_accuracy'])} | {fmt(d['defect_precision'])} | "
                 f"{fmt(d['defect_recall'])} | {fmt(d['decision_accuracy'])} | {d['length_mae_mm']:.2f} mm |")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors=INK2, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _cm(ax, labels_r, labels_c, counts, title, normalize=True):
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", BLUE_RAMP)
    M = np.array([[counts.get((r, c), 0) for c in labels_c] for r in labels_r], float)
    show = M / np.maximum(M.sum(axis=1, keepdims=True), 1) if normalize else M
    ax.imshow(show, cmap=cmap, vmin=0, vmax=1 if normalize else max(1, show.max()))
    ax.set_xticks(range(len(labels_c)), labels_c, rotation=35, ha="right", fontsize=9, color=INK2)
    ax.set_yticks(range(len(labels_r)), labels_r, fontsize=9, color=INK2)
    ax.set_xlabel("predicted", color=MUTED, fontsize=9)
    ax.set_ylabel("ground truth", color=MUTED, fontsize=9)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if M[i, j]:
                ax.text(j, i, f"{int(M[i, j])}", ha="center", va="center", fontsize=8,
                        color="#ffffff" if show[i, j] > 0.55 else INK)
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    for s in ax.spines.values():
        s.set_visible(False)


def plots(m, tc, dcm, cdef, fa, meas_err, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})

    fig, axs = plt.subplots(1, 2, figsize=(10, 4.2))
    _cm(axs[0], TYPES, TYPES, tc, "Glove type (row-normalised)")
    _cm(axs[1], DECISIONS, DECISIONS, dcm, "Decision vs. rule-table ground truth")
    fig.tight_layout()
    fig.savefig(out / "confusion_type_decision.png", dpi=130)
    plt.close(fig)

    cols = DEFECTS + ["missed"]
    rows = DEFECTS
    counts = dict(cdef)
    for t, v in fa.items():
        counts[("false alarm", t)] = v
    fig, ax = plt.subplots(figsize=(9, 7.2))
    _cm(ax, rows + ["false alarm"], cols, counts, "Defects: ground truth vs. detected type (row-normalised)")
    fig.tight_layout()
    fig.savefig(out / "confusion_defects.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 3.8))
    x = np.arange(len(DEFECTS))
    pr = [m["defects"][t]["precision"] for t in DEFECTS]
    rc = [m["defects"][t]["recall"] for t in DEFECTS]
    ax.bar(x - 0.19, pr, 0.36, color=S1, label="precision")
    ax.bar(x + 0.19, rc, 0.36, color=S2, label="recall")
    ax.set_xticks(x, [t.replace("_", "\n") for t in DEFECTS], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "25 %", "50 %", "75 %", "100 %"])
    ax.legend(frameon=False, fontsize=9, loc="lower left", ncols=2)
    ax.set_title("Per-defect precision and recall (synthetic test set)", loc="left", fontsize=11, color=INK)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out / "defect_precision_recall.png", dpi=130)
    plt.close(fig)

    types = [t for t in DEFECTS if t in meas_err]
    if types:
        fig, ax = plt.subplots(figsize=(9, 3.8))
        ax.boxplot([meas_err[t] for t in types], tick_labels=[t.replace("_", "\n") for t in types], widths=0.5,
                   showfliers=False, medianprops=dict(color=S2, linewidth=2),
                   boxprops=dict(color=S1), whiskerprops=dict(color=S1), capprops=dict(color=S1))
        ax.axhline(0, color="#c3c2b7", linewidth=1)
        ax.set_ylabel("measured − true (mm)", color=MUTED, fontsize=9)
        ax.set_title("Defect size measurement error (matched detections)", loc="left", fontsize=11, color=INK)
        _style(ax)
        fig.tight_layout()
        fig.savefig(out / "measurement_error.png", dpi=130)
        plt.close(fig)

    pl = m["per_lighting"]
    keys = [k for k in ["nominal", "dim", "bright", "warm_gradient", "shadow_noise"] if k in pl]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    x = np.arange(len(keys))
    series = [("type accuracy", "type_accuracy", S1), ("defect recall", "defect_recall", S2),
              ("decision accuracy", "decision_accuracy", S3)]
    for i, (name, key, col) in enumerate(series):
        ax.bar(x + (i - 1) * 0.26, [pl[k][key] for k in keys], 0.24, color=col, label=name)
    ax.set_xticks(x, [k.replace("_", " + ") if k == "shadow_noise" else k.replace("_", " ") for k in keys], fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "25 %", "50 %", "75 %", "100 %"])
    ax.legend(frameon=False, fontsize=9, loc="lower left", ncols=3)
    ax.set_title("Robustness to lighting (background learned under nominal light only)", loc="left", fontsize=11,
                 color=INK)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out / "lighting.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
