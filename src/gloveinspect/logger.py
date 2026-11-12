"""CSV logging: one row per glove (gloves.csv) and one row per defect (defects.csv)."""
from __future__ import annotations

import csv
import datetime as dt
from pathlib import Path

GLOVE_FIELDS = ["timestamp", "session", "source", "glove_id", "type", "type_confidence", "hand", "direction",
                "length_mm", "width_mm", "length_dev_mm", "width_dev_mm", "n_frames", "n_defects", "defect_types",
                "integrity", "cosmetic", "dimension", "quality_score", "decision"]
DEFECT_FIELDS = ["timestamp", "session", "glove_id", "defect", "size_mm", "x_glove_mm", "y_glove_mm", "details"]


class CsvLogger:
    def __init__(self, folder: str | Path, source: str = ""):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.session = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.source = source
        self.gloves_path = self.folder / f"gloves_{self.session}.csv"
        self.defects_path = self.folder / f"defects_{self.session}.csv"
        for p, f in ((self.gloves_path, GLOVE_FIELDS), (self.defects_path, DEFECT_FIELDS)):
            with open(p, "w", newline="", encoding="utf-8") as fh:
                csv.writer(fh).writerow(f)

    def log(self, rec):
        ts = dt.datetime.fromtimestamp(rec.timestamp).isoformat(timespec="milliseconds")
        d = rec.decision
        row = [ts, self.session, self.source, rec.id, rec.type, rec.type_conf, rec.hand, rec.direction,
               rec.length_mm, rec.width_mm, rec.length_dev_mm, rec.width_dev_mm, rec.n_obs, len(rec.defects),
               ";".join(sorted(x.type for x in rec.defects)), d.integrity, d.cosmetic, d.dimension, d.score, d.label]
        with open(self.gloves_path, "a", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow(row)
        with open(self.defects_path, "a", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            for x in rec.defects:
                det = ";".join(f"{k}={v}" for k, v in x.measures.items())
                w.writerow([ts, self.session, rec.id, x.type, round(x.size_mm, 2), round(x.gx_mm, 1), round(x.gy_mm, 1), det])
