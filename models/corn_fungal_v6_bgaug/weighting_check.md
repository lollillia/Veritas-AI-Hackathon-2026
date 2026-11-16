# Place + date weighting check: `models/corn_fungal_v6_bgaug`

Test photos of `data/splits_v6.csv` with a real city and capture date (`data/image_metadata.json`): **676**. Places found: 11 of 11; weather found for 40 of 40 city-days.

| | Photo only | Weighted |
|---|---|---|
| **Accuracy** | 95.4% | 95.1% |

Answers changed by weighting: 4 (wrong → right 1, right → wrong 3, wrong → other wrong 0).

| Source | Photos | Photo only | Weighted |
|---|---|---|---|
| cds | 157 | 96.2% | 96.2% |
| china_2 | 30 | 86.7% | 83.3% |
| kaggle | 2 | 100.0% | 100.0% |
| mendeley_bd | 116 | 98.3% | 97.4% |
| ndisan | 274 | 95.3% | 95.3% |
| neau | 85 | 96.5% | 96.5% |
| up_field | 12 | 75.0% | 75.0% |
