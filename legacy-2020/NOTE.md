# legacy-2020 – original 2020 code (unchanged)

These are my own files from the 2020 university group project on vision-based glove defect
inspection and sorting. They are kept **exactly as written in 2020**
for comparison with the 2026 redo in `../src`. Teammates' code from 2020 is not included.

| File | What it did |
|---|---|
| `defects.py` | Defect detection: fixed HSV range for a blue glove, contour area + extent rules → *Hole*, *Stain*; skin-colour mask → *Tear* (exposed fingertip) |
| `GUI.py` | Tkinter window with a button that starts the `defects.py` loop (the *Record* button refers to an undefined variable) |
| `test.py`, `test2.py`, `test3.py` | Trackbar experiments for tuning the HSV ranges |
| `glove.py`, `hull.py`, `markers.py` | Early contour / convex-hull / bounding-shape experiments on still images (the images are not included) |
| `Images/glove3.mp4`, `Images/glove4.mp4` | My 2020 test videos: a blue nitrile glove worn on a hand, on white paper |

## Running it in 2026

The scripts were written for Python 3.7, OpenCV 4.4 and NumPy 1.19. With current NumPy,
`np.int0` no longer exists (use NumPy < 2, or alias `np.int0 = np.intp`). `defects.py` creates
trackbars on a window that is never opened (its `namedWindow` call is commented out); newer
OpenCV versions raise an error for that.

What happened when I re-ran `defects.py` in 2026 (window calls stubbed out, nothing else changed):

| Video | Result |
|---|---|
| `glove4.mp4` (the video the thresholds were tuned on) | Hole, stain and exposed fingertip are marked in the right places, plus one false *Tear* on the wooden table |
| `glove3.mp4` (same glove type, different lighting/table) | 15–20 false *Tear* boxes per frame on the paper and table; the real holes are labelled *Tear*, *Hole* appears in 5 of 478 frames |

This is the main reason for the redo: fixed colour thresholds only worked for the one video they
were tuned on. See the main README for the 2026 approach.
