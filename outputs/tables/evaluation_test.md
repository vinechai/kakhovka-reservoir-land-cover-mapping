# Evaluation on `test` (297 hand-labelled points)

| | accuracy | macro-F1 | area-weighted accuracy | Lyman month-to-month change (km², median) |
|---|---|---|---|---|
| phase1 | 0.758 | 0.755 | 0.797 | 0.01 |
| cnn | 0.741 | 0.696 | 0.785 | 0.01 |

cnn minus phase 1, 95% bootstrap interval: macro-F1 [-0.107, -0.010], accuracy [-0.064, +0.030]

Per-class F1:

| class | phase1 | cnn |
|---|---|---|
| bare | 0.75 | 0.76 |
| dense_veg | 0.88 | 0.85 |
| sparse_veg | 0.53 | 0.35 |
| water | 0.86 | 0.83 |

Confusion, phase1 (rows = hand label):

| label      |   bare |   dense_veg |   sparse_veg |   water |
|:-----------|-------:|------------:|-------------:|--------:|
| bare       |     83 |           4 |           24 |       0 |
| dense_veg  |      0 |          68 |           10 |       1 |
| sparse_veg |     16 |           4 |           32 |       0 |
| water      |     10 |           0 |            3 |      42 |

Confusion, cnn (rows = hand label):

| label      |   bare |   dense_veg |   sparse_veg |   water |
|:-----------|-------:|------------:|-------------:|--------:|
| bare       |     92 |           6 |           11 |       2 |
| dense_veg  |      1 |          73 |            4 |       1 |
| sparse_veg |     25 |          13 |           14 |       0 |
| water      |     13 |           1 |            0 |      41 |

Area on the reservoir bed, 2024-08: estimated from the 191 points labelled for that month (95% interval), vs the maps (km²):

| class      |   km2 |   ci95_low |   ci95_high |   phase1 map |   cnn map |
|:-----------|------:|-----------:|------------:|-------------:|----------:|
| water      |  31.1 |       24.6 |        37.6 |         23.1 |      24.9 |
| bare       |  74.1 |       62.4 |        85.8 |         70   |      81.9 |
| sparse_veg |  28.9 |       19.6 |        38.2 |         32.6 |      20.3 |
| dense_veg  | 144   |      133   |       154.9 |        152.4 |     150.9 |