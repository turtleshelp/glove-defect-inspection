# Design proposal – Glove Defect Inspection & Sorting (2026 redo)

Status: **approved 12 Nov 2026 and built.** Measured results are in `../results/metrics.md`; the README
lists where they meet or miss the targets in §7.

**Decisions so far (12 Nov 2026):**
- Solo redo; no 2020 teammate code reused. The 2020 group is credited only as context.
- All visually detectable defect types (§4), not just 3.
- Data: synthetic generator first, then one real box of nitrile gloves for a real-world check.
- 3 glove types: **exam glove**, **long-cuff (chemical) glove**, **garden/work glove** (differ in size and shape).
- GUI: **PySide6**.
- Hardware: Arduino + 1 servo available → 3-position diverter.
- legacy-2020/: `defects.py`, `GUI.py`, `test*.py`, `glove.py`, `hull.py`, `markers.py`, `glove3.mp4`, `glove4.mp4`.

## 1. Scene & data

| Item | Choice | Why |
|---|---|---|
| Camera | Top-down, fixed height (any USB/laptop/phone camera, ≥ 720p) | Removes perspective error, makes mm measurement fair |
| Glove pose | Laid **flat** on a plain matte background (not worn) | Closer to a factory inspection line; a worn hand made "skin = defect" ambiguous in 2020 |
| Calibration | Printed checkerboard (known square size) → homography → mm/px | Replaces the fixed 0.2646 mm/px (96-dpi screen value) used in 2020 |
| Data | **Synthetic video generator** (exact ground truth) + **small real set** + 2020 videos (qualitative) | See §6 |

## 2. Pipeline

```
frame ─► illumination normalisation ─► segmentation ─► glove contour
          (gray-world WB + CLAHE on L)   (background model in LAB,      │
                                          ΔE + Otsu, morphology)        ▼
                                                     shape features (mm)
                                                     ├─► glove type  (MLP)
                                                     ├─► size check  (tolerance)
                                                     ├─► orientation (PCA + thumb)
                                                     └─► defect detectors ─► fuzzy decision ─► sort (GUI bins / servo)
                                                                                     │
                                                                              CSV log + GUI totals
```

**Segmentation is colour-agnostic:** the system learns the empty background at start-up and segments
whatever differs from it, so glove colour is never used for identification and
lighting drift is handled by normalisation + adaptive (Otsu) thresholds.

**Per-glove decision, not per-frame:** a glove is tracked while it is in view; it is judged once when
fully visible, using a vote over several frames, so totals count gloves, not frames.

## 3. Glove type, size and orientation

- **Shape features:** area, length (principal axis), palm width, aspect ratio, solidity, finger count
  and finger lengths (convexity defects), cuff length/width, 7 Hu moments (log-scaled).
- **Type (3 classes):** small MLP (scikit-learn) on these features. No colour, no template matching.
- **Size tolerance:** measured length/width vs. nominal per type in `config.yaml` (e.g. ±5 mm).
- **Orientation:**
  - *top/bottom* – which end of the principal axis has the fingers (convexity-defect valleys).
  - *left/right* – side of the thumb after rotating fingers-up.
  - **Known limit:** from a silhouette, a left glove palm-up looks identical to a right glove palm-down.
    The system reports L/R under a fixed convention (back of hand facing camera) and the README says so.

## 4. Defect catalogue (all types detectable with one top camera)

| # | Defect | Method | Stored measurement |
|---|---|---|---|
| 1 | Hole | Background visible inside glove mask (inner contours) | Equivalent diameter, area (mm, mm²) |
| 2 | Pinhole | Same as hole, small-size class | Diameter (mm); min. detectable size = f(mm/px), stated in README |
| 3 | Edge tear / notch | Deviation of contour from smoothed contour away from finger valleys | Tear length/depth (mm) |
| 4 | Missing / short finger | Finger count < 5, or finger length < x % of nominal for that type | Which finger, missing length (mm) |
| 5 | Fused / webbed fingers | Shallow finger valley depth / one over-wide finger | Valley depth (mm) |
| 6 | Stain / discolouration | Pixels inside glove with ΔE (LAB) far from the glove's median colour, larger blobs | Area (mm²), mean ΔE |
| 7 | Dirt / foreign particle | Small, high-contrast dark blobs inside glove | Size (mm), count |
| 8 | Cuff defect (torn / uneven / missing bead) | Straightness + width profile of the cuff edge | Max deviation (mm) |
| 9 | Dimensional out-of-tolerance | Length/width vs. nominal | Deviation (mm) |

**Not covered (needs other hardware), stated honestly in README:** thin spots and bubbles/blisters
(need backlight), internal contamination, inside-out gloves.

## 5. Decision & sorting

- **Fuzzy logic (scikit-fuzzy):** inputs = worst defect severity, total defect area, size deviation;
  output = quality score → **PASS / REWORK (second grade) / REJECT**. Rule base in config.
- So the project uses **both** a neural network (type) and fuzzy logic (decision).
- **Sorting:** three bins in the GUI; optional Arduino + 1 servo as a 3-position diverter
  (e.g. 30° / 90° / 150°). Serial protocol: one character per glove (`P`, `R`, `X`) + ACK.
  Runs without hardware (simulated bins only).

## 6. Data plan

**Synthetic generator (`src/glovesim/`)** – procedural, no AI-generated video:
- Parametric glove silhouette (palm, 5 finger capsules, cuff) for 3 types, left/right, random pose.
- Random glove colour (so colour cannot be used as a cue), wrinkle texture, background textures.
- Defects 1–9 injected with exact ground truth (position, mm size, type).
- Lighting variations: brightness ±40 %, colour temperature, gradient, shadows, noise, blur, JPEG.
- Output: conveyor-style video + JSON ground truth per glove.

**Real data:** synthetic results alone don't prove real-world performance. Cheapest real check:
one box of nitrile gloves (~€5–8), damage some on purpose (scissors, needle, marker, dirt),
measure them with a ruler, film them. The README reports synthetic and real results separately.

## 7. Test plan (targets, not results)

| Metric | Data | Target |
|---|---|---|
| Type accuracy (3 classes) | synthetic, 600 gloves, 5 lighting conditions | ≥ 95 % |
| Length / width error | synthetic (exact GT) / real (ruler) | MAE ≤ 2 mm / ≤ 3 mm |
| Hole diameter error | synthetic | MAE ≤ 0.5 mm for holes ≥ 2 mm |
| Per-defect precision & recall | synthetic, ≥ 60 instances per type | ≥ 90 % (holes, tears, missing/fused finger, stain); ≥ 80 % (pinhole, dirt, cuff) |
| Orientation accuracy | synthetic | ≥ 98 % |
| Decision confusion matrix | synthetic + real | reported, no target |
| Lighting robustness | same gloves, 5 lighting levels | accuracy drop ≤ 5 points |
| Speed | 720p, laptop CPU | ≥ 15 FPS |

Results that miss a target are reported as they are.

## 8. Repo layout

```
src/gloveinspect/   calibration, segmentation, features, classify, orientation,
                    defects/, decision (fuzzy), tracking, sorter (serial), logger
src/glovesim/       synthetic data generator
app/                GUI
firmware/           Arduino servo sketch
config/             config.yaml (calibration, thresholds, tolerances, fuzzy rules)
tests/              pytest unit tests
results/            evaluate.py, confusion matrices, plots
legacy-2020/        Ali's original 2020 files, untouched + NOTE.md
docs/  media/
```
