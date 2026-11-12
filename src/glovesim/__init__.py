"""glovesim – procedural generator of synthetic glove-inspection images and videos.

Every glove is drawn from a parametric model (cuff, palm, five fingers) in millimetres, defects are
cut or painted into it with exact ground truth, and the scene is rendered through a slightly tilted
camera with varying lighting. Nothing here is AI-generated; all randomness comes from a seeded
NumPy generator, so datasets are reproducible.
"""

from .model import GLOVE_TYPES, DEFECT_TYPES, GloveSpec, sample_glove, render_glove  # noqa: F401
