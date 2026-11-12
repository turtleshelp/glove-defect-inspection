import copy

import numpy as np
import pytest

from glovesim.model import GLOVE_TYPES, GloveSpec, render_glove, sample_glove
from glovesim.scene import LIGHTING, WORLD_H, WORLD_W, Scene
from gloveinspect import Inspector

BLUE = (0.75, 0.40, 0.10)


class Env:
    def __init__(self):
        self.rng = np.random.default_rng(123)
        self.scene = Scene(self.rng)
        nominal = LIGHTING["nominal"]
        self.calib = self.scene.finish(self.scene.to_camera(self.scene.calibration_world()), nominal)
        self.background = self.scene.finish(self.scene.to_camera(self.scene.plate.copy()), nominal)
        self.insp = Inspector()
        self.insp.calibrate(self.calib)
        self.insp.learn_background([self.background])

    def image(self, glove_type="exam", hand="R", defects=(), angle=10.0, lighting="nominal", seed=0, colour=BLUE,
              exact=False):
        rng = np.random.default_rng(seed)
        if exact:
            spec = GloveSpec(glove_type, hand, copy.deepcopy(GLOVE_TYPES[glove_type]), colour, list(defects))
        else:
            spec = sample_glove(rng, glove_type, hand, colour, list(defects))
        glove = render_glove(spec, rng, plate_bgr=self.scene.plate_bgr)
        M, _ = self.scene.glove_affine(glove, (WORLD_W / 2, WORLD_H / 2), angle)
        world, _ = self.scene.composite(self.scene.plate.copy(), glove, M)
        img = self.scene.finish(self.scene.to_camera(world), LIGHTING[lighting], np.random.default_rng(seed + 1))
        return img, glove

    def inspect(self, img):
        _, recs = self.insp.inspect_still(img)
        assert len(recs) == 1, f"expected one glove, got {len(recs)}"
        return recs[0]


@pytest.fixture(scope="session")
def env():
    return Env()
