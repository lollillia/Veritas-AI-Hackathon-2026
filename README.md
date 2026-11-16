# Verity Corn: detecting fungal diseases in corn

Upload a photo of a corn (maize) leaf and get one result: healthy or one of ten fungal diseases (common rust, southern rust, northern leaf blight, southern leaf blight, gray leaf spot, northern leaf spot, eyespot, Phaeosphaeria leaf spot, brown spot, Curvularia leaf spot), or "not a corn leaf", with a confidence and a one-line note. Built as a hackathon demo.

The website is the same whichever model runs behind it. `config.yaml` picks one of four interchangeable detectors:

| `detector:` | What it is | Needs |
|---|---|---|
| `gated` | Claude checks "is this a corn leaf?", then our fine-tuned model names the disease. If Claude is unreachable, the fine-tuned model answers alone | `ANTHROPIC_API_KEY`, `requirements-ml.txt`, a trained model (`models/corn_fungal_v6_bgaug`, included in the repo) |
| `llm_vision` | A vision LLM (Claude) via API | `ANTHROPIC_API_KEY`, internet |
| `hf_pretrained` | A pretrained corn-disease model from Hugging Face | `requirements-ml.txt`, a `model_id` and `label_map` in config |
| `finetuned` (current) | Our own model trained with `train/train.py`, alone. Needs no API keys; cannot answer "not a corn leaf" | `requirements-ml.txt`, a trained model (`models/corn_fungal_v6_bgaug`, included in the repo) |

## Setup

Requires **Python 3.12 or newer** (tested on 3.12 and 3.14). Check with `python3 --version`; if needed,
install a newer Python from [python.org](https://www.python.org/downloads/) or with `brew install python@3.14`.

```bash
python3 -m venv .venv && source .venv/bin/activate   # or python3.12 / python3.14 -m venv .venv
python --version                           # should print Python 3.12 or newer
python -m pip install -r requirements.txt            # web app + LLM backend
python -m pip install -r requirements-ml.txt         # the model (torch, transformers): needed by the default `finetuned` detector, `gated`, `hf_pretrained` and training
python -m pip install -r requirements-dev.txt        # only to run the tests (pytest, httpx; also installs requirements.txt)
```

### Secrets

Never put secrets in `config.yaml`; it only holds the *name* of an environment variable.

- `ANTHROPIC_API_KEY`: key for the `gated` and `llm_vision` backends. Not needed with the default `finetuned` detector.
- `HF_TOKEN`: only needed for private or gated Hugging Face models (public ones need none). `huggingface-cli login` also works.

Either `export` them in your shell (or set them in the PyCharm run configuration), or copy `.env.example` to `.env` and fill it in. `.env` is gitignored.

## Run

From the project folder, with the virtual environment active (`source .venv/bin/activate`):

```bash
python -m uvicorn app.main:build_app --factory --reload
# open http://127.0.0.1:8000
```

Use `python -m uvicorn`, not plain `uvicorn`: it always runs with the active environment's Python, so it finds the
installed packages. If the config is wrong (unknown detector, missing key, missing weights), startup stops with a
message naming the problem.

**`ModuleNotFoundError: No module named 'fastapi'` (or `transformers`)** means the Python running the app does not
have the packages:
1. Activate the environment in this terminal: `source .venv/bin/activate` (Windows: `.venv\Scripts\activate`).
2. Install into it: `python -m pip install -r requirements.txt -r requirements-ml.txt`.
3. Check: `python -c "import fastapi, transformers; print('ok')"`, then start with `python -m uvicorn ...` as above.

## Place, date and weather

The page also asks where (city, state/region, country, optional ZIP) and when (day, month, year) the photo was
taken. Both are optional.

1. The place is looked up on OpenStreetMap, and the weather for the 14 and 30 days before the date on Open-Meteo.
   Results are cached in `data/raw/app_cache/`; nothing else is stored.
2. Each disease's photo percentage is multiplied by a region, season and weather factor from
   `app/knowledge/diseases.yaml`. The factors are sourced, mild (×0.4–1.3 each) and capped at ×0.3–1.5 combined,
   so a clear photo still wins. `healthy` is never changed.
3. The page shows the final disease, a spectrum of the top diseases, the photo-only percentages, why the
   numbers moved, and fixed "About the disease" text with its sources.

The knowledge file was built from the user's source list (`app/knowledge/SOURCES.md` says which sources
were read, blocked or skipped). Those sources are mostly about the United States, so region factors exist
mainly for US states; elsewhere only season and weather apply. The facts come from outside sources, not
from our photos, which avoids the fingerprinting problem in
`train/issues/METADATA_FINGERPRINTING_ISSUE.md`.

**Measured effect** (`python -m train.weighting_check`, `models/corn_fungal_v6_bgaug/weighting_check.md`):
on 676 test photos with a real place and date, accuracy was 95.4% photo-only and 95.1% weighted. Weighting
changed 4 answers: 1 to right, 3 to wrong. Most of those photos are from places the knowledge file says little
about (Indonesia, Bangladesh, China), so the test says more about "does no harm" than about the benefit
on US field photos. The factors were not tuned on this result.

## Switching models

Edit `detector:` in `config.yaml` and restart. Low-confidence results (below `min_confidence`, default 0.5) are shown as "not sure".

### Using a Hugging Face model (`hf_pretrained`) — still to do

No model has been chosen yet; `config.yaml` has a placeholder `model_id`. To pick one:

1. Find 2–3 corn-disease image-classification models on Hugging Face and check their licenses.
2. Try each on at least 6 corn leaf photos (include one non-corn photo).
3. Set `hf_pretrained.model_id` in `config.yaml`, and map the model's own class names to ours in `label_map`, for example `Corn_(maize)___Common_rust_: common_rust`. Startup fails if any model class is left unmapped.

Note: a corn-only classifier has no "not a corn leaf" class, so it relies on the confidence threshold for that case.

### Training your own (`finetuned`)

The website uses `models/corn_fungal_v6_bgaug` (committed in the repo), trained on `data/splits_v6.csv` with all 11 labels and background augmentation: 91.3% on 1,719 held-out test photos, but about 74% on hard field photos (details in `models/corn_fungal_v6_bgaug/MODEL_PERFORMANCE.md`; comparison with other models in `models/MODEL_COMPARISON.md`). It was trained on corn leaves only, so on its own (`detector: finetuned`) it can never answer "not a corn leaf": in a check, 33 of 41 non-corn photos got a confident disease at the 0.5 threshold. The `gated` detector fixes that. In a live check, Claude rejected 41 of 41 non-corn photos (including 19 bean leaves) and accepted 59 of 60 corn test photos, taking about 2 seconds per photo. A history of all models is in `train/summaries/`.

Data sources, licenses and label mapping are in `docs/data-sources.md`. Every image's source, label and train/dev/test split is recorded in `data/splits.csv` and `data/splits_v2.csv` (committed; the images themselves are not). Never regenerate these: they record what each model trained on; add data with `--base`.

```bash
python -m pip install -r requirements-ml.txt
python -m data_prep.build_splits --add NAME=PATH --base data/splits_v2.csv --out data/splits_v3.csv   # add a source; existing splits stay put
python -m train.train --manifest data/splits.csv --out models/corn --epochs 5
python -m train.evaluate --manifest data/splits.csv --model models/corn --split test
```

Folder names are mapped to labels by `train/labels.py`; unknown folders (for example `Streak_virus`) are ignored. The older `--data <folder>` mode of `train.train` still works but makes its own random split.

## Retraining from the Hugging Face copy of the data

The raw images are not in git (about 30 GB). They are stored as a public Hugging Face dataset (https://huggingface.co/datasets/lollillia/veritas-corn-raw). One script
downloads them, trains a model, scores it and writes a report:

```bash
source .venv/bin/activate
scripts/train_from_hf.sh                             # dataset lollillia/veritas-corn-raw, model models/corn_retrained_YYYYMMDD
scripts/train_from_hf.sh lollillia/veritas-corn-raw corn_retrained   # or name them explicitly
```

No Hugging Face login is needed: `lollillia/veritas-corn-raw` is a public dataset. For a private dataset, run
`hf auth login` (or set `HF_TOKEN`) first.

- It downloads the dataset into `data/raw` (it handles either the folder or a single `raw.zip`) and checks
  that every image in `data/splits_v6.csv` is present. `CHECK_ONLY=1` stops there; `SKIP_DOWNLOAD=1` uses
  images already on disk.
- It trains MobileNetV2 with background augmentation (`EPOCHS=5` by default; `STD_AUGMENT=1` adds random
  crop/flip/colour changes), about 40-80 minutes on an Apple-silicon Mac.
- It scores the model on the test photos, runs the background-swap test and the place-and-date check, and
  writes `models/<name>/REPORT.md`, compared with the website's current model, with a recommendation.
- It never overwrites an existing model folder.

**To enable the new model in the website**, follow the last section of the report. In short, set
`finetuned.weights_path: models/<name>` in `config.yaml` (with `detector: finetuned`) and restart with
`python -m uvicorn app.main:build_app --factory`. To go back, point `weights_path` at the previous folder.

## Adding another detector

1. Add `app/detectors/<name>.py` with a class that subclasses `Detector` and implements `predict(image_bytes) -> Result`.
2. Add one line to `DETECTOR_PATHS` in `app/detectors/registry.py`.
3. Add a section named `<name>` to `config.yaml`; its keys are passed to your constructor.

## Tests

```bash
python -m pip install -r requirements-dev.txt
pytest
```

## Two-minute demo script

1. "Farmers lose yield to leaf diseases they often spot too late." Open the page.
2. Drop in a diseased leaf. Show the label, confidence bar and note.
3. Drop in a photo that is not a corn leaf. Show it says so instead of guessing.
4. Open `config.yaml`, change `detector:`, restart, and run the same photo. "The website never changes; the model behind it is a plug-in."

## Resources

Related work (papers, code) and the external services the data pipeline uses are listed in [`docs/resources.md`](docs/resources.md).
