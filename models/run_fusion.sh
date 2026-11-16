#!/bin/zsh
cd /Users/chun_han/Workspace/git/github/hackthon/Veritas
for split in v3 v3_by_date; do
  for mode in image weather; do
    flag=""; [[ $mode == weather ]] && flag="--with-weather"
    echo "=== $split $mode $(date +%H:%M:%S)"
    .venv/bin/python -m train.fusion --manifest data/splits_$split.csv --out models/fusion_${split}_${mode} --epochs 5 $flag 2>&1 | grep -v -E "Warning|warn|NOTE|Loading weights|UNEXPECTED|can be ignored"
  done
done
echo "=== ALL DONE $(date +%H:%M:%S)"
