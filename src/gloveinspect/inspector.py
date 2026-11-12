"""Inspector: ties calibration, segmentation, per-glove analysis, tracking, voting and the decision together.

Video / live use:  inspector.process(frame) every frame -> FrameResult (with finished GloveRecords)
Single image use:  inspector.inspect_still(frame)       -> list of GloveRecords
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from .analysis import Defect, GloveObservation, analyse_glove
from .calibration import Rectifier
from .classify import TypeClassifier, load_golden
from .config import Config, load_config
from .decision import Decision, FuzzyDecider
from .segmentation import BackgroundModel, Segmentation

LOOSE_TYPES = {"missing_finger", "fused_fingers", "cuff_defect", "out_of_tolerance"}


@dataclass
class GloveRecord:
    id: int
    timestamp: float
    type: str
    type_conf: float
    hand: str
    direction: str
    length_mm: float
    width_mm: float
    length_dev_mm: float
    width_dev_mm: float
    defects: list
    decision: Decision
    n_obs: int
    contour: np.ndarray | None = None
    observation: GloveObservation | None = None


@dataclass
class Blob:
    contour: np.ndarray
    complete: bool
    track_id: int | None = None
    obs: GloveObservation | None = None


@dataclass
class FrameResult:
    rect: np.ndarray
    seg: Segmentation | None
    blobs: list = field(default_factory=list)
    finished: list = field(default_factory=list)
    learning: bool = False


@dataclass
class _Track:
    id: int
    center: tuple
    obs: list = field(default_factory=list)
    missed: int = 0
    done: bool = False
    last_contour: np.ndarray | None = None
    seen_complete: int = 0


class Inspector:
    def __init__(self, cfg: Config | None = None, rectifier: Rectifier | None = None):
        self.cfg = cfg or load_config()
        self.rect = rectifier
        self.bgm: BackgroundModel | None = None
        specs = self.cfg["glove_types"]
        self.classifier = TypeClassifier.load(self.cfg.path(self.cfg["models"]["type_classifier"]), specs)
        self.golden = load_golden(self.cfg.path(self.cfg["models"]["golden"]))
        dcfg = self.cfg["decision"]
        self.decider = FuzzyDecider(dcfg["pass_min"], dcfg["rework_min"])
        self._learn = []
        self.reset_tracking()

    # ------------------------------------------------------------------ setup
    def calibrate(self, calib_img: np.ndarray) -> Rectifier:
        c = self.cfg["calibration"]
        b = c["board"]
        self.rect = Rectifier.from_checkerboard(calib_img, b["cols"], b["rows"], b["square_mm"], c["output_px_per_mm"])
        self.bgm = None
        return self.rect

    def use_uncalibrated(self, shape):
        self.rect = Rectifier.uncalibrated(shape, self.cfg["calibration"]["fallback_px_per_mm"])
        self.bgm = None

    def learn_background(self, frames: list[np.ndarray]):
        if self.rect is None:
            self.use_uncalibrated(frames[0].shape)
        self.bgm = BackgroundModel(self.cfg["segmentation"], self.rect.px_per_mm)
        self.bgm.learn([self.rect.warp(f) for f in frames])

    def start_background_learning(self):
        self._learn = []
        self.bgm = None

    def reset_tracking(self):
        self._tracks: list[_Track] = []
        self._next_id = 1

    @property
    def ready(self) -> bool:
        return self.rect is not None and self.bgm is not None and self.bgm.ready

    # ------------------------------------------------------------------ per frame
    def _segment(self, frame):
        rect = self.rect.warp(frame)
        valid = self.rect.valid_mask(frame.shape)
        seg = self.bgm.segment(rect, valid)
        inner = cv2.erode(valid, np.ones((9, 9), np.uint8))
        blobs = []
        for c in seg.contours:
            pts = c.reshape(-1, 2)
            complete = bool(np.all(inner[pts[:, 1], pts[:, 0]] > 0))
            blobs.append(Blob(c, complete))
        return rect, valid, seg, blobs

    def _analyse(self, seg, contour):
        try:
            return analyse_glove(seg, contour, self.rect.px_per_mm, self.cfg, self.classifier, self.golden)
        except Exception as e:  # never crash the live loop on one odd blob
            print("analysis failed:", type(e).__name__, e)
            return None

    def process(self, frame: np.ndarray, analyse: bool = True) -> FrameResult:
        """One video frame. Learns the background first (needs an empty plate), then inspects."""
        if self.rect is None:
            self.use_uncalibrated(frame.shape)
        if not self.ready:
            self._learn.append(frame)
            n = self.cfg["segmentation"]["learn_frames"]
            if len(self._learn) >= n:
                self.learn_background(self._learn)
                self._learn = []
            return FrameResult(self.rect.warp(frame), None, learning=True)
        rect, valid, seg, blobs = self._segment(frame)
        ppm = self.rect.px_per_mm
        tcfg = self.cfg["tracking"]
        # --- associate blobs with tracks (nearest centroid)
        centers = []
        for b in blobs:
            m = cv2.moments(b.contour)
            centers.append((m["m10"] / m["m00"], m["m01"] / m["m00"]))
        used = set()
        for tr in self._tracks:
            best, bd = None, tcfg["max_jump_mm"] * ppm
            for i, c in enumerate(centers):
                if i in used:
                    continue
                d = np.hypot(c[0] - tr.center[0], c[1] - tr.center[1])
                if d < bd:
                    best, bd = i, d
            if best is None:
                tr.missed += 1
            else:
                used.add(best)
                tr.center = centers[best]
                tr.missed = 0
                blobs[best].track_id = tr.id
                tr.last_contour = blobs[best].contour
        for i, b in enumerate(blobs):
            if b.track_id is None:
                tr = _Track(self._next_id, centers[i], last_contour=b.contour)
                self._next_id += 1
                self._tracks.append(tr)
                b.track_id = tr.id
        # --- analyse complete blobs of open tracks
        by_id = {t.id: t for t in self._tracks}
        finished = []
        for b in blobs:
            tr = by_id[b.track_id]
            if tr.done:
                continue
            if b.complete and analyse:
                tr.seen_complete += 1
                every = tcfg.get("analyse_every", 1)
                if (tr.seen_complete - 1) % every == 0 and len(tr.obs) < tcfg.get("max_observations", 999):
                    b.obs = self._analyse(seg, b.contour)
                    if b.obs is not None:
                        tr.obs.append(b.obs)
                        if len(tr.obs) >= tcfg.get("max_observations", 999):
                            finished.append(self._finalize(tr))  # enough evidence: decide now, sort early
            elif not b.complete and len(tr.obs) >= tcfg["min_observations"]:
                finished.append(self._finalize(tr))
        for tr in list(self._tracks):
            if tr.missed >= 2:
                if not tr.done and len(tr.obs) >= 1:
                    finished.append(self._finalize(tr))
                self._tracks.remove(tr)
        self.bgm.update(seg, valid)
        return FrameResult(rect, seg, blobs, finished)

    def flush(self) -> list[GloveRecord]:
        """End of a video: finalise gloves still in view."""
        out = [self._finalize(t) for t in self._tracks if not t.done and t.obs]
        self._tracks = []
        return out

    def inspect_still(self, frame: np.ndarray) -> tuple[FrameResult, list[GloveRecord]]:
        rect, valid, seg, blobs = self._segment(frame)
        recs = []
        for b in blobs:
            if not b.complete:
                continue
            b.obs = self._analyse(seg, b.contour)
            if b.obs is None:
                continue
            tr = _Track(self._next_id, b.obs.center, [b.obs], last_contour=b.contour)
            self._next_id += 1
            b.track_id = tr.id
            recs.append(self._finalize(tr))
        return FrameResult(rect, seg, blobs, recs), recs

    # ------------------------------------------------------------------ voting
    def _finalize(self, tr: _Track) -> GloveRecord:
        tr.done = True
        obs = tr.obs
        n = len(obs)
        tcfg = self.cfg["tracking"]
        votes: dict[str, float] = {}
        for o in obs:
            for k, v in (o.type_probs or {o.type: 1.0}).items():
                votes[k] = votes.get(k, 0.0) + v
        gtype = max(votes, key=votes.get)
        conf = votes[gtype] / n
        hands = [o.hand for o in obs if o.hand != "?"]
        hand = max(set(hands), key=hands.count) if hands else "?"
        dirs = [o.direction for o in obs]
        direction = max(set(dirs), key=dirs.count)
        length = float(np.median([o.length_mm for o in obs]))
        width = float(np.median([o.width_mm for o in obs]))
        spec = self.cfg["glove_types"].get(gtype, {"length": length, "width": width})
        tol = self.cfg["tolerance"]
        dl, dw = length - spec["length"], width - spec["width"]

        # cluster defects across frames in the glove frame
        clusters: list[dict] = []
        r_tight = tcfg["defect_merge_mm"]
        for k, o in enumerate(obs):
            for d in o.defects:
                if d.type == "out_of_tolerance":
                    continue
                r = 30.0 if d.type in LOOSE_TYPES else r_tight
                hit = None
                for cl in clusters:
                    if cl["type"] == d.type and np.hypot(cl["gx"] - d.gx_mm, cl["gy"] - d.gy_mm) < r:
                        hit = cl
                        break
                if hit is None:
                    clusters.append(dict(type=d.type, gx=d.gx_mm, gy=d.gy_mm, items=[(k, d)]))
                else:
                    hit["items"].append((k, d))
                    hit["gx"] = float(np.mean([it[1].gx_mm for it in hit["items"]]))
                    hit["gy"] = float(np.mean([it[1].gy_mm for it in hit["items"]]))
        last = obs[-1]
        defects = []
        for cl in clusters:
            frames = {k for k, _ in cl["items"]}
            if len(frames) < max(1, tcfg["vote_fraction"] * n):
                continue
            items = sorted(cl["items"], key=lambda it: it[1].size_mm)
            rep = items[len(items) // 2][1]
            # position: prefer the detection from the last frame (for drawing)
            in_last = [d for k, d in cl["items"] if k == n - 1]
            pos = in_last[0] if in_last else rep
            defects.append(Defect(rep.type, pos.x, pos.y, rep.gx_mm, rep.gy_mm, rep.size_mm,
                                  dict(rep.measures, seen=f"{len(frames)}/{n}")))
        if abs(dl) > tol["length"] or abs(dw) > tol["width"]:
            size = max(abs(dl) - tol["length"], abs(dw) - tol["width"])
            defects.append(Defect("out_of_tolerance", last.center[0], last.center[1], 0.0, length / 2, size,
                                  dict(length_dev_mm=round(dl, 1), width_dev_mm=round(dw, 1))))
        decision = self.decider.decide(defects, dl, dw, tol)
        return GloveRecord(tr.id, time.time(), gtype, round(conf, 3), hand, direction, round(length, 1), round(width, 1),
                           round(dl, 1), round(dw, 1), defects, decision, n, tr.last_contour, last)
