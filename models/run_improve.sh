#!/bin/zsh
# Field-photo improvements: standard augmentation, unseen-dataset hold-out (up_field), (stronger backbone skipped).
cd "$(dirname "$0")/.."
PY=.venv/bin/python
M=data/splits_v6.csv
score() {   # $1 model dir
  $PY -m train.evaluate --manifest $M --model models/$1 --split test --json models/$1/eval_v6.json > models/$1/eval_v6.txt 2>&1
  $PY -m train.bg_swap --model models/$1 --pool unseen --json models/$1/bg_swap_unseen.json > models/$1/bg_swap_unseen.txt 2>&1
}
holdout() { # $1 model dir: score on every up_field photo (none were trained on)
  $PY -m train.evaluate --manifest $M --model models/$1 --split all --source up_field --json models/$1/eval_up_field_all.json > models/$1/eval_up_field_all.txt 2>&1
}
echo "A start $(date)"
$PY -m train.train --manifest $M --out models/corn_fungal_v6_aug2 --epochs 5 --bg-augment --std-augment > models/train_fungal_v6_aug2.log 2>&1 && score corn_fungal_v6_aug2
echo "B start $(date)"
$PY -m train.train --manifest $M --out models/holdout_upfield_bgaug --epochs 5 --bg-augment --exclude-source up_field > models/train_holdout_upfield_bgaug.log 2>&1 && holdout holdout_upfield_bgaug
echo "C start $(date)"
$PY -m train.train --manifest $M --out models/holdout_upfield_aug2 --epochs 5 --bg-augment --std-augment --exclude-source up_field > models/train_holdout_upfield_aug2.log 2>&1 && holdout holdout_upfield_aug2
echo "done $(date)"
