#!/bin/zsh
# v6: baseline and background-augmented MobileNetV2, then test-set scores on v6 and on v5's test photos.
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python
$PY -m train.train --manifest data/splits_v6.csv --out models/corn_fungal_v6 --epochs 5 > models/train_fungal_v6.log 2>&1
$PY -m train.train --manifest data/splits_v6.csv --out models/corn_fungal_v6_bgaug --epochs 5 --bg-augment > models/train_fungal_v6_bgaug.log 2>&1
for m in corn_fungal_v6 corn_fungal_v6_bgaug; do
  $PY -m train.evaluate --manifest data/splits_v6.csv --model models/$m --split test --json models/$m/eval_v6.json > models/$m/eval_v6.txt 2>&1
  $PY -m train.evaluate --manifest data/splits_v5.csv --model models/$m --split test --json models/$m/eval_on_v5_test.json > models/$m/eval_on_v5_test.txt 2>&1
done
echo done
