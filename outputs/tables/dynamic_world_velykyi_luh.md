# Dynamic World vs our maps, `velykyi_luh` reservoir bed

## Dynamic World classes on the bed (km²)

|                    |   2024-08 |   2025-08 |   2026-08 |
|:-------------------|----------:|----------:|----------:|
| bare               |      30.3 |      18.6 |      13.7 |
| built              |      19.7 |       3.8 |       3.1 |
| crops              |     131   |      31.6 |       6.2 |
| flooded_vegetation |       1.8 |       8.5 |       7.4 |
| grass              |       0.4 |       5.8 |       7.8 |
| shrub_and_scrub    |       1.9 |      32.7 |      13.9 |
| snow_and_ice       |       0   |       0.1 |       0.2 |
| trees              |      47.4 |     149.1 |     195.7 |
| water              |      45.6 |      27.8 |      30.1 |

## 2024-08: what Dynamic World calls each `baseline` class (row shares)

| baseline   |   bare |   built |   crops |   flooded_vegetation |   grass |   shrub_and_scrub |   snow_and_ice |   trees |   water |
|:-----------|-------:|--------:|--------:|---------------------:|--------:|------------------:|---------------:|--------:|--------:|
| bare       |     39 |       9 |      27 |                    0 |       0 |                 1 |              0 |       0 |      23 |
| dense_veg  |      0 |       5 |      62 |                    1 |       0 |                 0 |              0 |      30 |       2 |
| nodata     |     43 |       1 |      56 |                    0 |       0 |                 0 |              0 |       0 |       0 |
| sparse_veg |      7 |      17 |      56 |                    1 |       0 |                 3 |              0 |       4 |      12 |
| water      |      0 |       0 |       1 |                    0 |       0 |                 0 |              0 |       0 |      99 |

## 2025-08: what Dynamic World calls each `baseline` class (row shares)

| baseline   |   bare |   built |   crops |   flooded_vegetation |   grass |   shrub_and_scrub |   snow_and_ice |   trees |   water |
|:-----------|-------:|--------:|--------:|---------------------:|--------:|------------------:|---------------:|--------:|--------:|
| bare       |     45 |       3 |      11 |                    2 |       0 |                26 |              0 |       2 |      10 |
| dense_veg  |      0 |       1 |      10 |                    3 |       3 |                 4 |              0 |      78 |       2 |
| sparse_veg |      3 |       3 |      23 |                    5 |       1 |                40 |              0 |      20 |       6 |
| water      |      0 |       0 |       0 |                    0 |       0 |                 0 |              0 |       0 |     100 |

## 2026-08: what Dynamic World calls each `baseline` class (row shares)

| baseline   |   bare |   built |   crops |   flooded_vegetation |   grass |   shrub_and_scrub |   snow_and_ice |   trees |   water |
|:-----------|-------:|--------:|--------:|---------------------:|--------:|------------------:|---------------:|--------:|--------:|
| bare       |     51 |       5 |       4 |                    2 |       0 |                21 |              1 |       3 |      13 |
| dense_veg  |      0 |       0 |       1 |                    2 |       4 |                 1 |              0 |      89 |       2 |
| sparse_veg |      3 |       5 |       9 |                    8 |       2 |                30 |              0 |      32 |      11 |
| water      |      0 |       0 |       0 |                    0 |       0 |                 0 |              0 |       0 |     100 |

## 2024-08: what Dynamic World calls each `cnn` class (row shares)

| cnn        |   bare |   built |   crops |   flooded_vegetation |   grass |   shrub_and_scrub |   snow_and_ice |   trees |   water |
|:-----------|-------:|--------:|--------:|---------------------:|--------:|------------------:|---------------:|--------:|--------:|
| bare       |     33 |      11 |      31 |                    0 |       0 |                 1 |              0 |       1 |      22 |
| dense_veg  |      0 |       5 |      61 |                    1 |       0 |                 0 |              0 |      30 |       2 |
| nodata     |     43 |       1 |      56 |                    0 |       0 |                 0 |              0 |       0 |       0 |
| sparse_veg |     11 |      16 |      59 |                    1 |       0 |                 2 |              0 |       8 |       3 |
| water      |      0 |       0 |       4 |                    0 |       0 |                 0 |              0 |       0 |      96 |

## 2025-08: what Dynamic World calls each `cnn` class (row shares)

| cnn        |   bare |   built |   crops |   flooded_vegetation |   grass |   shrub_and_scrub |   snow_and_ice |   trees |   water |
|:-----------|-------:|--------:|--------:|---------------------:|--------:|------------------:|---------------:|--------:|--------:|
| bare       |     27 |       3 |      16 |                    3 |       1 |                33 |              0 |       7 |       9 |
| dense_veg  |      0 |       1 |      11 |                    3 |       3 |                 5 |              0 |      76 |       2 |
| sparse_veg |     11 |       3 |      19 |                    7 |       2 |                17 |              0 |      39 |       1 |
| water      |      0 |       0 |       0 |                    1 |       0 |                 0 |              0 |       0 |      99 |

## 2026-08: what Dynamic World calls each `cnn` class (row shares)

| cnn        |   bare |   built |   crops |   flooded_vegetation |   grass |   shrub_and_scrub |   snow_and_ice |   trees |   water |
|:-----------|-------:|--------:|--------:|---------------------:|--------:|------------------:|---------------:|--------:|--------:|
| bare       |     32 |       5 |       6 |                    4 |       1 |                26 |              1 |      12 |      13 |
| dense_veg  |      0 |       0 |       2 |                    3 |       3 |                 1 |              0 |      88 |       2 |
| sparse_veg |      5 |       3 |       8 |                    7 |       3 |                 8 |              0 |      63 |       3 |
| water      |      0 |       0 |       0 |                    1 |       0 |                 0 |              0 |       0 |      99 |

## Hand labels vs Dynamic World (121 points)

| hand label   |   bare |   built |   crops |   flooded_vegetation |   shrub_and_scrub |   trees |   water |
|:-------------|-------:|--------:|--------:|---------------------:|------------------:|--------:|--------:|
| bare         |      9 |      10 |      24 |                    0 |                 0 |       1 |       4 |
| dense_veg    |      0 |       3 |      19 |                    1 |                 0 |      17 |       1 |
| sparse_veg   |      2 |       2 |       8 |                    0 |                 1 |       3 |       0 |
| water        |      0 |       0 |       1 |                    0 |                 0 |       0 |      15 |
