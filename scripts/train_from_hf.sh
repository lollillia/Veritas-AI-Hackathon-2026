#!/usr/bin/env bash
# Download the raw images from Hugging Face, train a model, evaluate it and write a report.
#
#   scripts/train_from_hf.sh [HF_REPO] [MODEL_NAME]
#
#   HF_REPO      the Hugging Face dataset (default: lollillia/veritas-corn-raw)
#   MODEL_NAME   folder name under models/ (default: corn_retrained_YYYYMMDD)
#
# Options (environment variables):
#   EPOCHS=5          training epochs
#   STD_AUGMENT=0     1 adds random crop/flip/colour augmentation (slower; see models/MODEL_COMPARISON.md)
#   SKIP_DOWNLOAD=0   1 uses the images already in data/raw
#   MANIFEST=data/splits_v6.csv
#   BASELINE=models/corn_fungal_v6_bgaug   the model to compare against (the website's current model)
#   CHECK_ONLY=0      1 stops after downloading and checking the images (no training)
#
# Needs: the virtual environment with requirements.txt + requirements-ml.txt installed. The default dataset is
# public, so no Hugging Face login is needed; for a private dataset, run `hf auth login` or set HF_TOKEN first.
# Training takes about 40-80 minutes on an Apple-silicon Mac.
set -euo pipefail

HF_REPO="${1:-lollillia/veritas-corn-raw}"
NAME="${2:-corn_retrained_$(date +%Y%m%d)}"
EPOCHS="${EPOCHS:-5}"
STD_AUGMENT="${STD_AUGMENT:-0}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
MANIFEST="${MANIFEST:-data/splits_v6.csv}"
BASELINE="${BASELINE:-models/corn_fungal_v6_bgaug}"
CHECK_ONLY="${CHECK_ONLY:-0}"

cd "$(dirname "$0")/.."
if [ -n "${VIRTUAL_ENV:-}" ]; then PY=python; HF=hf; elif [ -x .venv/bin/python ]; then PY=.venv/bin/python; HF=.venv/bin/hf; else PY=python3; HF=hf; fi
OUT="models/$NAME"
step() { printf '\n=== %s ===\n' "$*"; }


if [ -e "$OUT/model.safetensors" ]; then
  echo "models/$NAME already exists; choose another MODEL_NAME so no trained model is overwritten." >&2
  exit 1
fi

# 1. Download --------------------------------------------------------------------------------------------
if [ "$SKIP_DOWNLOAD" != "1" ]; then
  step "Downloading $HF_REPO from Hugging Face"
  # The upload may be the data/raw folder itself or a single raw.zip; handle both.
  if "$PY" -c "import sys; from huggingface_hub import list_repo_files as f; sys.exit(0 if 'raw.zip' in f(sys.argv[1], repo_type='dataset') else 1)" "$HF_REPO"; then
    "$HF" download "$HF_REPO" raw.zip --repo-type dataset --local-dir data
    step "Unzipping data/raw.zip"
    unzip -q -n data/raw.zip -d data          # -n: never overwrite files already in data/raw
  else
    "$HF" download "$HF_REPO" --repo-type dataset --local-dir data/raw
  fi
fi

# The folder data/raw/china_2 was renamed to data/raw/IDADP; uploads made before that still use the old name.
if [ -d data/raw/china_2 ] && [ ! -e data/raw/IDADP ]; then
  echo "Renaming data/raw/china_2 to data/raw/IDADP (folder renamed in the project)"
  mv data/raw/china_2 data/raw/IDADP
fi

# 2. Check every image in the manifest is present ---------------------------------------------------------
step "Checking the images listed in $MANIFEST"
"$PY" - "$MANIFEST" <<'EOF'
import csv, sys
from pathlib import Path
rows = list(csv.DictReader(open(sys.argv[1])))
missing = [r["path"] for r in rows if r["split"] in ("train", "dev", "test") and not Path(r["path"]).is_file()]
print(f"{len(rows) - len(missing)} of {len(rows)} images present")
if missing:
    print("missing, for example:", *missing[:5], sep="\n  ")
    sys.exit(f"{len(missing)} images are missing; check the download")
EOF
if [ "$CHECK_ONLY" = "1" ]; then echo "CHECK_ONLY=1: images are in place; stopping before training."; exit 0; fi

# 3. Leaf masks for background augmentation ----------------------------------------------------------------
step "Computing leaf masks (cached in data/raw/mask_cache)"
"$PY" -m data_prep.leaf_masks --manifest "$MANIFEST"

# 4. Train -------------------------------------------------------------------------------------------------
AUG="--bg-augment"; DESC="background augmentation"
if [ "$STD_AUGMENT" = "1" ]; then AUG="--bg-augment --std-augment"; DESC="background + standard augmentation"; fi
step "Training $OUT ($EPOCHS epochs, $DESC)"
mkdir -p "$OUT"
# AUG holds one or two flags on purpose, so it is not quoted
"$PY" -m train.train --manifest "$MANIFEST" --out "$OUT" --epochs "$EPOCHS" $AUG 2>&1 | tee "$OUT/train.log"

# 5. Evaluate ----------------------------------------------------------------------------------------------
step "Scoring on the test photos"
"$PY" -m train.evaluate --manifest "$MANIFEST" --model "$OUT" --split test --json "$OUT/eval_v6.json" | tee "$OUT/eval_v6.txt"
step "Background-swap test"
"$PY" -m train.bg_swap --model "$OUT" --pool unseen --json "$OUT/bg_swap_unseen.json" | tee "$OUT/bg_swap_unseen.txt"
step "Place and date weighting check (needs internet; skipped if it fails)"
"$PY" -m train.weighting_check --model "$OUT" --manifest "$MANIFEST" --out "$OUT/weighting_check.md" \
  || echo "weighting check skipped (no internet or lookup failure); the report will leave it out"

# 6. Report ------------------------------------------------------------------------------------------------
step "Writing $OUT/REPORT.md"
"$PY" -m train.report --model "$OUT" --baseline "$BASELINE" >/dev/null
sed -n '/^## Recommendation/,$p' "$OUT/REPORT.md"
echo
echo "Full report: $OUT/REPORT.md"
