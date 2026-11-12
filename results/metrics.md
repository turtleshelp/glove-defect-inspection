# Evaluation – synthetic test set (900 gloves)

All numbers are on **synthetic** images (generated with a different seed than the training set). They are not real-world results.

| Glove-level metric | Value |
|---|---|
| Gloves found | 900 / 900 |
| Glove type accuracy (3 classes) | 99.9 % |
| Left/right hand accuracy | 97.7 % (undecided: 1.8 %) |
| Top/bottom (finger direction) accuracy | 100.0 % |
| Length error | MAE 0.80 mm, 95th pct 1.25 mm |
| Width error | MAE 0.71 mm, 95th pct 1.09 mm |
| Defective gloves flagged (any defect) | 97.5 % |
| Good gloves wrongly flagged | 2.7 % |
| PASS/REWORK/REJECT decision accuracy | 95.1 % |
| Processing time per image (CPU) | 513 ms mean |

| Defect | GT count | Precision | Recall | F1 | Size error (MAE) |
|---|---|---|---|---|---|
| hole | 85 | 93.1 % | 95.3 % | 94.2 % | 0.29 mm (bias -0.22) |
| pinhole | 82 | 79.3 % | 89.0 % | 83.9 % | 0.28 mm (bias +0.24) |
| tear | 68 | 92.3 % | 88.2 % | 90.2 % | 4.16 mm (bias -4.16) |
| missing_finger | 88 | 68.6 % | 81.8 % | 74.6 % | 2.06 mm (bias +0.13) |
| fused_fingers | 88 | 75.9 % | 100.0 % | 86.3 % | – |
| stain | 90 | 91.6 % | 96.7 % | 94.1 % | 1.04 mm (bias +0.31) |
| dirt | 167 | 97.2 % | 82.6 % | 89.3 % | 0.38 mm (bias +0.33) |
| cuff_defect | 75 | 77.7 % | 97.3 % | 86.4 % | 0.66 mm (bias -0.21) |
| out_of_tolerance | 80 | 88.9 % | 100.0 % | 94.1 % | – |

| Lighting | n | Type acc. | Hand acc. | Defect precision | Defect recall | Decision acc. | Length MAE |
|---|---|---|---|---|---|---|---|
| nominal | 180 | 100.0 % | 97.8 % | 91.3 % | 97.4 % | 98.9 % | 0.51 mm |
| dim | 180 | 100.0 % | 98.9 % | 89.3 % | 94.4 % | 97.8 % | 0.57 mm |
| bright | 180 | 100.0 % | 98.3 % | 94.0 % | 93.4 % | 96.1 % | 0.58 mm |
| warm_gradient | 180 | 100.0 % | 97.2 % | 88.5 % | 93.6 % | 97.8 % | 0.62 mm |
| shadow_noise | 180 | 99.4 % | 96.1 % | 65.7 % | 79.1 % | 85.0 % | 1.70 mm |
