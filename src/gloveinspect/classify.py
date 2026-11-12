"""Glove-type classification (small MLP on shape features) and learned "golden" shape statistics."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

FEATURES = ["length_mm", "area_mm2", "max_width_mm", "aspect", "solidity", "extent",
            "w50_mm", "w65_mm", "w80_mm", "w95_mm", "hu1", "hu2", "hu3"]


def feature_vector(feats: dict) -> np.ndarray:
    return np.array([feats[k] for k in FEATURES], np.float64)


class TypeClassifier:
    """Wraps a scikit-learn pipeline (StandardScaler + MLPClassifier).

    Without a trained model it falls back to the nearest nominal length (so the pipeline still runs).
    """

    def __init__(self, model=None, specs: dict | None = None):
        self.model = model
        self.specs = specs or {}

    @classmethod
    def load(cls, path: str | Path, specs: dict | None = None) -> "TypeClassifier":
        p = Path(path)
        if not p.exists():
            return cls(None, specs)
        import joblib
        return cls(joblib.load(p), specs)

    def save(self, path: str | Path):
        import joblib
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)

    @property
    def trained(self) -> bool:
        return self.model is not None

    def predict(self, feats: dict):
        if self.model is None:
            if not self.specs:
                return "unknown", 0.0, {}
            t = min(self.specs, key=lambda k: abs(self.specs[k]["length"] - feats["length_mm"]))
            return t, 0.0, {t: 1.0}
        p = self.model.predict_proba(feature_vector(feats)[None, :])[0]
        classes = list(self.model.classes_)
        k = int(np.argmax(p))
        return classes[k], float(p[k]), {c: float(v) for c, v in zip(classes, p)}

    @staticmethod
    def train(X: np.ndarray, y: list[str], seed: int = 0):
        from sklearn.neural_network import MLPClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        model = make_pipeline(StandardScaler(),
                              MLPClassifier(hidden_layer_sizes=(32, 16), alpha=1e-3, max_iter=3000, random_state=seed))
        model.fit(X, y)
        return TypeClassifier(model)


def learn_golden(stats_by_type: dict[str, list[dict]]) -> dict:
    """Median shape statistics of defect-free gloves, per type."""
    out = {}
    for t, items in stats_by_type.items():
        good = [s for s in items if s["n_finger_peaks"] == 4 and s["thumb_tip"] is not None]
        if len(good) < 5:
            continue
        g = dict(
            finger_len_ratio=np.median([s["finger_len"] for s in good], axis=0).round(4).tolist(),
            tip_s_ratio=np.median([s["finger_s"] for s in good], axis=0).round(4).tolist(),
            thumb_asym=round(float(np.median([s["thumb_asym"] for s in good])), 4),
            thumb_tip=np.median([s["thumb_tip"] for s in good], axis=0).round(4).tolist(),
            valley_elev_mm=round(float(np.median([v for s in good for v in s["valley_elev"]] or [0.0])), 2),
            palm_top_ratio=round(float(np.median([s["palm_top_ratio"] for s in good])), 4),
            n=len(good),
        )
        out[t] = g
    return out


def load_golden(path: str | Path) -> dict:
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else {}
