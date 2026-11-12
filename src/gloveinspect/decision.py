"""Fuzzy-logic quality decision: PASS / REWORK / REJECT.

Three crisp inputs are computed from the measured defects:
  integrity  0..10  barrier defects (holes, pinholes, tears, missing/fused fingers, cuff damage)
  cosmetic   0..10  stains and dirt
  dimension  0..3   size deviation in multiples of the tolerance
A Mamdani system (scikit-fuzzy) maps them to a quality score 0..100 (centroid defuzzification).
The rule base mirrors a typical medical-glove policy: any barrier defect rejects the glove, cosmetic
or size problems send it to rework / second grade, otherwise it passes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

INTEGRITY = {"hole", "pinhole", "tear", "missing_finger", "fused_fingers", "cuff_defect"}
COSMETIC = {"stain", "dirt"}


@dataclass
class Decision:
    label: str
    score: float
    integrity: float
    cosmetic: float
    dimension: float


def severities(defects, length_dev_mm: float, width_dev_mm: float, tol: dict):
    integ = 0.0
    cos = 0.0
    for d in defects:
        t, s = d.type, d.size_mm
        if t == "hole":
            integ = max(integ, 6 + s / 3)
        elif t == "pinhole":
            integ = max(integ, 5 + s)
        elif t == "tear":
            integ = max(integ, 5 + s / 3)
        elif t == "missing_finger":
            integ = max(integ, 8)
        elif t == "fused_fingers":
            integ = max(integ, 6)
        elif t == "cuff_defect":
            integ = max(integ, 4 + s / 2)
        elif t == "stain":
            cos += 3 + s / 5
        elif t == "dirt":
            cos += 2.5
    dim = max(abs(length_dev_mm) / tol["length"], abs(width_dev_mm) / tol["width"])
    return min(integ, 10.0), min(cos, 10.0), min(dim, 3.0)


class FuzzyDecider:
    def __init__(self, pass_min: float = 65, rework_min: float = 35):
        import skfuzzy as fuzz
        from skfuzzy import control as ctrl

        self.pass_min, self.rework_min = pass_min, rework_min
        integ = ctrl.Antecedent(np.arange(0, 10.01, 0.05), "integrity")
        cos = ctrl.Antecedent(np.arange(0, 10.01, 0.05), "cosmetic")
        dim = ctrl.Antecedent(np.arange(0, 3.01, 0.01), "dimension")
        q = ctrl.Consequent(np.arange(0, 100.1, 0.5), "quality")
        integ["none"] = fuzz.trapmf(integ.universe, [0, 0, 0.5, 2])
        integ["minor"] = fuzz.trimf(integ.universe, [1, 3, 5])
        integ["major"] = fuzz.trapmf(integ.universe, [4, 6, 10, 10])
        cos["none"] = fuzz.trapmf(cos.universe, [0, 0, 0.5, 1.5])
        cos["light"] = fuzz.trimf(cos.universe, [0.5, 3, 6])
        cos["heavy"] = fuzz.trapmf(cos.universe, [5, 7, 10, 10])
        dim["ok"] = fuzz.trapmf(dim.universe, [0, 0, 0.8, 1.0])
        dim["off"] = fuzz.trapmf(dim.universe, [0.9, 1.1, 3, 3])
        q["reject"] = fuzz.trapmf(q.universe, [0, 0, 20, 40])
        q["rework"] = fuzz.trimf(q.universe, [30, 50, 70])
        q["pass"] = fuzz.trapmf(q.universe, [60, 80, 100, 100])
        rules = [
            ctrl.Rule(integ["major"], q["reject"]),
            ctrl.Rule(integ["minor"], q["rework"]),
            ctrl.Rule(integ["none"] & cos["none"] & dim["ok"], q["pass"]),
            ctrl.Rule(integ["none"] & cos["light"], q["rework"]),
            ctrl.Rule(integ["none"] & cos["heavy"], q["rework"]),
            ctrl.Rule(integ["none"] & dim["off"], q["rework"]),
        ]
        self.system = ctrl.ControlSystem(rules)
        self.membership = dict(integrity=integ, cosmetic=cos, dimension=dim, quality=q)

    def score(self, integ: float, cos: float, dim: float) -> float:
        from skfuzzy import control as ctrl
        sim = ctrl.ControlSystemSimulation(self.system)
        sim.input["integrity"] = integ
        sim.input["cosmetic"] = cos
        sim.input["dimension"] = dim
        try:
            sim.compute()
            return float(sim.output["quality"])
        except Exception:  # no rule fired - should not happen with these overlapping sets
            return 0.0 if integ >= 4 else (50.0 if cos > 0.5 or dim > 1 else 85.0)

    def decide(self, defects, length_dev_mm: float, width_dev_mm: float, tol: dict) -> Decision:
        integ, cos, dim = severities(defects, length_dev_mm, width_dev_mm, tol)
        s = self.score(integ, cos, dim)
        label = "PASS" if s >= self.pass_min else ("REWORK" if s >= self.rework_min else "REJECT")
        return Decision(label, round(s, 1), round(integ, 2), round(cos, 2), round(dim, 2))


def rule_table_decision(defect_types) -> str:
    """Crisp reference policy used as ground truth for the evaluation."""
    ts = set(defect_types)
    if ts & INTEGRITY:
        return "REJECT"
    if ts & (COSMETIC | {"out_of_tolerance"}):
        return "REWORK"
    return "PASS"
