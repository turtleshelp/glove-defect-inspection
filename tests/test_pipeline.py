"""End-to-end tests on synthetic gloves rendered in memory (fixed seeds, deterministic)."""
import numpy as np
import pytest

from glovesim.scene import WORLD_PPM


def test_calibration_scale(env):
    r = env.insp.rect
    assert r.rms_mm < 0.3
    # two world points 300 mm apart -> camera -> rectified: distance must be 300 mm
    pw = np.array([[400.0, 500.0], [400.0 + 300 * WORLD_PPM, 500.0]])
    pr = r.img_to_rect(env.scene.world_to_cam(pw))
    d_mm = np.linalg.norm(pr[1] - pr[0]) / r.px_per_mm
    assert abs(d_mm - 300.0) < 1.0


@pytest.mark.parametrize("gtype", ["exam", "longcuff", "garden"])
def test_clean_glove_measured_and_passes(env, gtype):
    img, glove = env.image(gtype, exact=True)
    rec = env.inspect(img)
    assert rec.type == gtype
    assert abs(rec.length_mm - glove["length_mm"]) < 2.0
    assert abs(rec.width_mm - glove["width_mm"]) < 2.0
    assert rec.defects == []
    assert rec.decision.label == "PASS"


@pytest.mark.parametrize("hand", ["R", "L"])
@pytest.mark.parametrize("angle,direction", [(15.0, "fingers_up"), (200.0, "fingers_down")])
def test_orientation(env, hand, angle, direction):
    img, _ = env.image("exam", hand=hand, angle=angle, exact=True)
    rec = env.inspect(img)
    assert rec.hand == hand
    assert rec.direction == direction


@pytest.mark.parametrize("lighting", ["dim", "bright", "warm_gradient", "shadow_noise"])
def test_lighting_robustness(env, lighting):
    img, glove = env.image("exam", lighting=lighting, exact=True, seed=3)
    rec = env.inspect(img)
    assert rec.type == "exam"
    assert abs(rec.length_mm - glove["length_mm"]) < 3.0


@pytest.mark.parametrize("defect,seed", [
    ("hole", 1), ("pinhole", 2), ("tear", 3), ("missing_finger", 4), ("fused_fingers", 5),
    ("stain", 6), ("dirt", 7), ("cuff_defect", 18), ("out_of_tolerance", 9),
])
def test_single_defect_detected(env, defect, seed):
    img, glove = env.image("exam", defects=[defect], seed=seed)
    rec = env.inspect(img)
    found = {d.type for d in rec.defects}
    assert defect in found, f"{defect} not found, got {found}"
    expected = "REWORK" if defect in ("stain", "dirt", "out_of_tolerance") else "REJECT"
    assert rec.decision.label == expected


def test_hole_diameter_measured(env):
    img, glove = env.image("exam", defects=["hole"], seed=11)
    rec = env.inspect(img)
    gt = [g for g in glove["gt"] if g["type"] == "hole"][0]
    holes = [d for d in rec.defects if d.type == "hole"]
    assert holes and abs(holes[0].size_mm - gt["diameter_mm"]) < 1.5
