"""Unit tests for the decision, logging, sorting and tracking parts."""
import csv

import pytest

from gloveinspect.analysis import Defect
from gloveinspect.decision import FuzzyDecider, rule_table_decision
from gloveinspect.logger import GLOVE_FIELDS, CsvLogger
from gloveinspect.sorter import SerialSorter, SimSorter

TOL = {"length": 10.0, "width": 6.0}


def D(t, size):
    return Defect(t, 0, 0, 0, 0, size, {})


@pytest.fixture(scope="module")
def fz():
    return FuzzyDecider(65, 35)


def test_fuzzy_basic_policy(fz):
    assert fz.decide([], 1.0, 0.5, TOL).label == "PASS"
    assert fz.decide([D("stain", 10)], 0, 0, TOL).label == "REWORK"
    assert fz.decide([D("dirt", 2)], 0, 0, TOL).label == "REWORK"
    assert fz.decide([], 20.0, 0, TOL).label == "REWORK"
    for t in ("hole", "pinhole", "tear", "missing_finger", "fused_fingers", "cuff_defect"):
        assert fz.decide([D(t, 5)], 0, 0, TOL).label == "REJECT", t
    assert fz.decide([D("hole", 6), D("stain", 10)], 0, 0, TOL).label == "REJECT"


def test_fuzzy_score_monotonic(fz):
    clean = fz.decide([], 0, 0, TOL).score
    stain_small = fz.decide([D("stain", 6)], 0, 0, TOL).score
    hole = fz.decide([D("hole", 10)], 0, 0, TOL).score
    assert clean > stain_small > hole


def test_rule_table():
    assert rule_table_decision([]) == "PASS"
    assert rule_table_decision(["stain"]) == "REWORK"
    assert rule_table_decision(["stain", "hole"]) == "REJECT"
    assert rule_table_decision(["out_of_tolerance"]) == "REWORK"


class _Rec:
    def __init__(self):
        from gloveinspect.decision import Decision
        self.id, self.timestamp, self.type, self.type_conf = 7, 1.7e9, "exam", 0.99
        self.hand, self.direction, self.length_mm, self.width_mm = "R", "fingers_up", 237.0, 95.0
        self.length_dev_mm, self.width_dev_mm, self.n_obs = -0.2, 0.0, 5
        self.defects = [Defect("hole", 1, 2, 3.0, 4.0, 6.5, {"diameter_mm": 6.5})]
        self.decision = Decision("REJECT", 15.0, 8.2, 0.0, 0.02)


def test_csv_logger(tmp_path):
    lg = CsvLogger(tmp_path, "unit")
    lg.log(_Rec())
    rows = list(csv.DictReader(open(lg.gloves_path, encoding="utf-8")))
    assert list(rows[0].keys()) == GLOVE_FIELDS
    assert rows[0]["decision"] == "REJECT" and rows[0]["defect_types"] == "hole"
    drows = list(csv.DictReader(open(lg.defects_path, encoding="utf-8")))
    assert drows[0]["defect"] == "hole" and float(drows[0]["size_mm"]) == 6.5


class FakeSerial:
    def __init__(self, port, baud, timeout=None):
        self.port, self.written, self.is_open = port, [], True

    def write(self, b):
        self.written.append(b)

    def readline(self):
        return b"OK X 150\n"

    def close(self):
        self.is_open = False


def test_serial_sorter_protocol():
    s = SerialSorter("COM9", 9600, serial_factory=FakeSerial)
    reply = s.sort(1, "REJECT")
    s.sort(2, "PASS")
    assert s.ser.written == [b"X\n", b"P\n"]
    assert reply == "OK X 150"
    assert s.bins == {"PASS": 1, "REWORK": 0, "REJECT": 1}


def test_sim_sorter():
    s = SimSorter()
    for lab in ("PASS", "REWORK", "REWORK"):
        s.sort(0, lab)
    assert s.bins == {"PASS": 1, "REWORK": 2, "REJECT": 0}
