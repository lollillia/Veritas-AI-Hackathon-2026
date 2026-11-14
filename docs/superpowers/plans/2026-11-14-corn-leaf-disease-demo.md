# Veritas Corn Leaf Disease Detector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A FastAPI website where a user uploads a corn leaf photo and gets one result (label, confidence, note) from a detector chosen in `config.yaml`.

**Architecture:** A `Detector` interface (`predict(bytes) -> Result`) with three plug-in backends (Hugging Face pretrained, locally fine-tuned, vision LLM) registered by name. `config.yaml` selects the backend; it is validated at startup. `create_app(detector, min_confidence)` builds the FastAPI app so tests can inject a fake detector. One static HTML page is the UI.

**Tech Stack:** Python 3.10+, FastAPI, uvicorn, Pillow, PyYAML, python-dotenv, anthropic SDK; transformers + torch (optional ML extras); pytest + httpx for tests.

**Spec:** `docs/superpowers/specs/2026-11-14-corn-leaf-disease-demo-design.md`

## Global Constraints

- Python only; no Node or build step.
- Labels are exactly: `healthy`, `gray_leaf_spot`, `common_rust`, `northern_leaf_blight`, `not_corn_leaf`.
- `Result` = `label`, `confidence` (float 0–1), `note` (one short sentence).
- Config file is `config.yaml`; detector names are `hf_pretrained`, `finetuned`, `llm_vision`; also `min_confidence` (default 0.5).
- Secrets are never in `config.yaml` or committed; config holds only env var names (e.g. `api_key_env`). HF token is `HF_TOKEN`; LLM key is `ANTHROPIC_API_KEY`. `.env` is gitignored; `.env.example` has placeholders only.
- A bad config name or a missing required secret/weights fails at startup with a message naming the problem.
- Uploads: JPEG/PNG/WebP, max 10 MB.
- Only the config file selects the model; one result per upload; no accounts, history or database.
- LLM default model id: `claude-sonnet-5-5`.
- Adding a fourth backend must mean adding one file plus one registry entry.

## Review Focus

Each line below has a test in the task named in brackets.

1. A corrupt or non-image file uploaded as `leaf.jpg` gets a clear 4xx error, not a 500. [Task 3]
2. An upload over 10 MB, and an empty or missing file, get clear errors. [Task 3]
3. RGBA/PNG-with-alpha, CMYK and EXIF-rotated photos are normalized to RGB before reaching any model. [Task 1]
4. The vision LLM returns malformed JSON, markdown-fenced JSON or an unknown label; the user sees a clear error instead of a crash or a made-up label. [Task 5]
5. A Hugging Face model whose labels are not covered by `label_map` fails at startup, not on the first upload. [Task 6]

---

### Task 1: Scaffold, core types, registry, image normalization

**Files:**
- Delete: `main.py` (PyCharm sample)
- Create: `.gitignore`, `.env.example`, `requirements.txt`, `requirements-ml.txt`, `requirements-dev.txt`, `app/__init__.py`, `app/detectors/__init__.py`, `app/detectors/base.py`, `app/detectors/registry.py`, `app/imaging.py`, `tests/__init__.py`, `tests/test_registry.py`, `tests/test_imaging.py`

**Interfaces:**
- Produces, in `app/detectors/base.py`:
  - `LABELS: tuple[str, ...]` (the five labels, in the order listed in Global Constraints)
  - `@dataclass(frozen=True) class Result: label: str; confidence: float; note: str`, which raises `ValueError` in `__post_init__` if the label is not in `LABELS` or the confidence is outside 0–1
  - `class Detector(ABC)` with abstract `predict(self, image_bytes: bytes) -> Result`
  - `class DetectorError(Exception)`: a user-presentable backend failure
- Produces, in `app/detectors/registry.py`:
  - `DETECTOR_PATHS: dict[str, str]` mapping name to `"module:Class"` (lazy import so choosing `llm_vision` never imports torch); initial entries for `hf_pretrained` (`app.detectors.hf_pretrained:HfPretrainedDetector`), `finetuned` (`app.detectors.finetuned:FinetunedDetector`), `llm_vision` (`app.detectors.llm_vision:LlmVisionDetector`)
  - `build_detector(name: str, settings: dict) -> Detector`: imports the class and calls `cls(**settings)`; raises `KeyError` with the message `unknown detector '<name>'` for an unknown name
- Produces, in `app/imaging.py`:
  - `load_rgb_image(image_bytes: bytes) -> PIL.Image.Image`: applies EXIF orientation, converts to RGB, raises `ValueError("not a valid image")` if undecodable
  - `to_jpeg_bytes(image: PIL.Image.Image, max_side: int = 1568) -> bytes`: downsizes if larger than `max_side`, returns JPEG bytes

- [ ] **Step 1: Initialize the repo.** Run `git init`; delete `main.py`; write `.gitignore` (must include `.env`, `models/`, `__pycache__/`, `.venv/`, `.idea/`); write `.env.example` with `HF_TOKEN=` and `ANTHROPIC_API_KEY=` placeholders; write the three requirements files (core: fastapi, uvicorn, python-multipart, pillow, pyyaml, python-dotenv, anthropic; ml: transformers, torch, datasets; dev: pytest, httpx); create a venv and install core + dev.
- [ ] **Step 2: Write failing tests.**
  - `tests/test_imaging.py`: `test_rgba_png_becomes_rgb` (assert `.mode == "RGB"`); `test_cmyk_jpeg_becomes_rgb`; `test_exif_rotation_applied` (a 20x10 JPEG with EXIF orientation 6 loads as size `(10, 20)`); `test_garbage_bytes_raise_value_error`; `test_to_jpeg_bytes_downsizes` (a 4000x2000 image gives a max side of 1568 and bytes starting with `b"\xff\xd8"`). Build images in the tests with Pillow.
  - `tests/test_registry.py`: `test_unknown_detector_raises_keyerror` (message contains `unknown detector 'nope'`); `test_result_rejects_unknown_label`; `test_result_rejects_confidence_above_one`.
- [ ] **Step 3: Run `pytest -v`; expect FAIL** (modules not found).
- [ ] **Step 4: Implement** `base.py`, `registry.py`, `imaging.py` to the interfaces above (`ImageOps.exif_transpose` for orientation).
- [ ] **Step 5: Run `pytest -v`; expect PASS.**
- [ ] **Step 6: Commit** `feat: scaffold project, detector interface, registry, image normalization`.

---

### Task 2: Config loading and fail-fast validation

**Files:**
- Create: `app/config.py`, `config.yaml`, `tests/test_config.py`

**Interfaces:**
- Consumes: `DETECTOR_PATHS` from `app/detectors/registry.py`
- Produces, in `app/config.py`:
  - `class ConfigError(Exception)`
  - `@dataclass(frozen=True) class Config: detector: str; settings: dict; min_confidence: float`
  - `load_config(path: str = "config.yaml", env: Mapping[str, str] | None = None) -> Config`: `env` defaults to `os.environ`. `settings` is the sub-dict under the detector's own key, passed to `build_detector` unchanged. Validation (each raises `ConfigError` with a message naming the problem):
    - the file is missing or not valid YAML
    - `detector` is missing or not in `DETECTOR_PATHS`
    - `llm_vision`: the env var named by `api_key_env` is unset (the message names that variable)
    - `finetuned`: `weights_path` does not exist
    - `min_confidence` is outside 0–1 (default 0.5)
- Produces: a default `config.yaml` at the repo root with `detector: llm_vision` and the example sections from the spec (`llm_vision.model: claude-sonnet-5-5`, `api_key_env: ANTHROPIC_API_KEY`).

- [ ] **Step 1: Write failing tests** in `tests/test_config.py`, using `tmp_path` YAML files and an explicit `env` dict: `test_valid_llm_config_loads`; `test_unknown_detector_names_it` (message contains the bad name); `test_missing_api_key_names_env_var` (message contains `ANTHROPIC_API_KEY`); `test_finetuned_missing_weights_path`; `test_min_confidence_out_of_range`; `test_missing_file`; `test_min_confidence_defaults_to_half`.
- [ ] **Step 2: Run `pytest tests/test_config.py -v`; expect FAIL.**
- [ ] **Step 3: Implement** `load_config` (`yaml.safe_load`; no secret values ever stored in `Config`).
- [ ] **Step 4: Run `pytest -v`; expect PASS.**
- [ ] **Step 5: Commit** `feat: config loading with fail-fast validation`.

---

### Task 3: FastAPI app and /predict endpoint

**Files:**
- Create: `app/api.py`, `app/main.py`, `static/index.html` (placeholder with a `<form>`; real UI in Task 4), `tests/test_api.py`

**Interfaces:**
- Consumes: `Detector`, `Result`, `DetectorError` (Task 1); `load_rgb_image` (Task 1); `load_config`, `build_detector` (Tasks 1–2)
- Produces, in `app/api.py`:
  - `create_app(detector: Detector, min_confidence: float = 0.5) -> FastAPI`: serves `static/index.html` at `GET /` and handles `POST /predict` (multipart field `file`)
  - Success response JSON: `{"label": str, "confidence": float, "note": str, "not_sure": bool}`; `not_sure` is true when `confidence < min_confidence` and the label is not `not_corn_leaf`
  - Errors return `{"error": "<message>"}`:
    - 400: missing or empty file
    - 413: over 10 MB
    - 415: not JPEG/PNG/WebP, judged by decoding with Pillow and checking `image.format`, not by the filename or content-type header
    - 502: the detector raised `DetectorError` (the message is passed through); any other exception also gives 502 with a generic message
- Produces, in `app/main.py`: `build_app(config_path: str = "config.yaml") -> FastAPI`, which calls `load_dotenv()`, then `load_config(config_path)`, then `build_detector(config.detector, config.settings)`, then `create_app(detector, config.min_confidence)`. There is no module-level `app`; run it with `uvicorn app.main:build_app --factory`. `ConfigError` propagates so startup fails with its message.

- [ ] **Step 1: Write failing tests** in `tests/test_api.py` using `TestClient` and a `FakeDetector` returning a fixed `Result` (the Task 1 pattern for building test images):
  - `test_predict_returns_result_json` (all four keys, `not_sure is False`)
  - `test_low_confidence_sets_not_sure` (confidence 0.3, min 0.5)
  - `test_not_corn_leaf_is_never_not_sure`
  - `test_corrupt_file_named_jpg_returns_415` (bytes `b"not an image"`, filename `leaf.jpg`, error message non-empty)
  - `test_text_file_returns_415`
  - `test_gif_returns_415`
  - `test_oversized_upload_returns_413` (just over 10 MB)
  - `test_empty_file_returns_400`
  - `test_missing_file_field_returns_400`
  - `test_detector_error_returns_502_with_message`
  - `test_unexpected_exception_returns_502_generic`
  - `test_index_page_served` (GET `/` is 200 and `text/html`)
- [ ] **Step 2: Run `pytest tests/test_api.py -v`; expect FAIL.**
- [ ] **Step 3: Implement** `create_app`. Read at most 10 MB + 1 bytes to detect oversize, then decode with `load_rgb_image` for validation and format check. Pass the original bytes to the detector.
- [ ] **Step 4: Run `pytest -v`; expect PASS.**
- [ ] **Step 5: Commit** `feat: FastAPI app with validated /predict endpoint`.

---

### Task 4: Upload UI

**Files:**
- Modify: `static/index.html` (replace the placeholder)

**Interfaces:**
- Consumes: `POST /predict` JSON contract from Task 3 (`label`, `confidence`, `note`, `not_sure`, or `{"error"}`)

One self-contained HTML file with inline CSS and JS, no CDN or build step. Required behavior:
- A drag-and-drop zone that also opens a file picker; accepts only JPEG/PNG/WebP and shows the preview.
- A spinner while waiting; the upload button is disabled during the request.
- A result card with a human-readable label (e.g. "Common rust"), a confidence bar with a percentage, and the note.
- Three distinct visual states: normal result, "not sure" (`not_sure` true) and "doesn't look like a corn leaf" (`not_corn_leaf`). Error responses show the `error` text in the card.
- Custom, polished styling (not browser defaults); usable at phone width. This is a judge demo, so the styling is the goal.

- [ ] **Step 1: Implement** the page as specified.
- [ ] **Step 2: Verify automatically:** `pytest tests/test_api.py::test_index_page_served -v`; expect PASS.
- [ ] **Step 3: Verify manually** with a temporary fake detector: run `python -c "import uvicorn; from app.api import create_app; from app.detectors.base import *; ..."` using a one-off `FakeDetector` (not committed), open the page, and upload (a) a real photo, (b) a text file renamed `.jpg`. Expected: the result card appears for (a), and a readable error appears for (b), with no console errors.
- [ ] **Step 4: Commit** `feat: upload UI with result card`.

---

### Task 5: Vision LLM detector

**Files:**
- Create: `app/detectors/llm_vision.py`, `tests/test_llm_vision.py`

**Interfaces:**
- Consumes: `Detector`, `Result`, `DetectorError`, `LABELS` (Task 1); `load_rgb_image`, `to_jpeg_bytes` (Task 1)
- Produces: `class LlmVisionDetector(Detector)` with `__init__(self, model: str = "claude-sonnet-5-5", api_key_env: str = "ANTHROPIC_API_KEY", client=None)`. When `client` is None, build `anthropic.Anthropic(api_key=os.environ[api_key_env])`. `predict` normalizes the image with `to_jpeg_bytes`, sends it as a base64 `image/jpeg` block via `client.messages.create`, and parses the reply.

The prompt must: say the image may not be a corn leaf; list the five exact labels; require JSON only: `{"label": ..., "confidence": 0-1, "note": "<one sentence>"}`; and require `not_corn_leaf` when the image is not a corn/maize leaf.

Parsing rules, in `_parse_reply(text: str) -> Result` (module-level for testing): strip optional markdown code fences; `json.loads`; unknown label, missing keys, non-numeric or out-of-range confidence, or non-JSON each raise `DetectorError("The model returned an unreadable answer; please try again.")`. API exceptions (`anthropic.APIError`) become `DetectorError("The vision service is unavailable; please try again.")`.

- [ ] **Step 1: Write failing tests** with a fake client object whose `messages.create` returns a canned reply (`.content[0].text`):
  - `test_parses_plain_json_reply` (label `common_rust`, 0.9)
  - `test_parses_fenced_json_reply` (reply wrapped in a ```` ```json ```` fence)
  - `test_unknown_label_raises_detector_error`
  - `test_malformed_json_raises_detector_error`
  - `test_confidence_out_of_range_raises_detector_error`
  - `test_api_error_becomes_detector_error`
  - `test_request_contains_jpeg_image_block_and_all_labels` (inspect the kwargs the fake client received)
  - `test_not_corn_leaf_reply_passes_through`
- [ ] **Step 2: Run `pytest tests/test_llm_vision.py -v`; expect FAIL.**
- [ ] **Step 3: Implement** `LlmVisionDetector` and `_parse_reply`.
- [ ] **Step 4: Run `pytest -v`; expect PASS.**
- [ ] **Step 5: Commit** `feat: vision LLM detector`.

---

### Task 6: Hugging Face pretrained detector

**Files:**
- Create: `app/detectors/hf_pretrained.py`, `tests/test_hf_pretrained.py`

**Interfaces:**
- Consumes: Task 1 types and `load_rgb_image`
- Produces: `class HfPretrainedDetector(Detector)` with `__init__(self, model_id: str, label_map: dict[str, str] | None = None, pipeline=None)`. When `pipeline` is None, build `transformers.pipeline("image-classification", model=model_id)` (the `HF_TOKEN` env var is picked up by the library). `label_map` maps the model's own labels to our five labels; None means identity.
  - Startup validation in `__init__`: every label in `pipeline.model.config.id2label.values()` must map (via `label_map`, or identity) to a member of `LABELS`. Otherwise raise `ValueError` listing the unmapped model labels.
  - `predict`: normalize with `load_rgb_image`, call `pipeline(image, top_k=1)`, map the top label, and return `Result(label, score, note)` with a fixed note per label (a dict of one-sentence descriptions for the four disease and healthy labels). Wrap pipeline exceptions in `DetectorError`.
  - Note: a corn-only classifier has no "not a corn leaf" class; low-confidence handling in the API layer covers that case.

- [ ] **Step 1: Write failing tests** with a `FakePipeline` (has `.model.config.id2label` and `__call__` returning `[{"label": ..., "score": ...}]`):
  - `test_maps_model_label_to_our_label` (e.g. `"Corn_Common_rust"` mapped to `common_rust`)
  - `test_identity_mapping_when_no_label_map`
  - `test_unmapped_model_label_fails_at_construction` (message lists the unmapped label)
  - `test_rgba_input_reaches_pipeline_as_rgb` (the fake records the image mode)
  - `test_pipeline_exception_becomes_detector_error`
  - `test_note_is_nonempty_for_every_label`
- [ ] **Step 2: Run `pytest tests/test_hf_pretrained.py -v`; expect FAIL.**
- [ ] **Step 3: Implement** the class.
- [ ] **Step 4: Run `pytest -v`; expect PASS.**
- [ ] **Step 5: Choose the real model.** Install `requirements-ml.txt`. Find 2–3 candidate corn-disease models on Hugging Face. Check each license, then run each on at least 6 sample corn leaf photos (include 1 non-corn photo). Pick one, fill in `hf_pretrained.model_id` and `label_map` in `config.yaml` (leave `detector: llm_vision` as the default), and record the choice and observed accuracy in `README.md` (Task 8).
- [ ] **Step 6: Commit** `feat: Hugging Face pretrained detector`.

---

### Task 7: Fine-tuned detector and training script

**Files:**
- Create: `app/detectors/finetuned.py`, `train/__init__.py`, `train/train.py`, `train/labels.py`, `tests/test_finetuned.py`, `tests/test_train_labels.py`

**Interfaces:**
- Consumes: `HfPretrainedDetector` (Task 6)
- Produces:
  - `class FinetunedDetector(HfPretrainedDetector)` with `__init__(self, weights_path: str, label_map: dict[str, str] | None = None, pipeline=None)`. `weights_path` is a local directory saved in Hugging Face format (by `save_pretrained`). It is the same as `HfPretrainedDetector(model_id=weights_path, ...)`; no duplicated prediction code.
  - `train/labels.py`: `folder_to_label(folder_name: str) -> str | None` maps common public-dataset folder names (PlantVillage style, e.g. `Corn_(maize)___Common_rust_`, `Corn_(maize)___Northern_Leaf_Blight`, `Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot`, `Corn_(maize)___healthy`) to our labels, returning None for folders that are not corn.
  - `train/train.py`: CLI `python -m train.train --data <imagefolder dir> --out models/corn --epochs 3 [--base-model <hf id>]` that fine-tunes a small pretrained image classifier on the folders `folder_to_label` recognizes, with `id2label` set to our label names, and saves it with `save_pretrained`. The default base model is a small one (e.g. a MobileNet or ConvNeXt-tiny) so training is feasible on a laptop or a free GPU notebook.

- [ ] **Step 1: Write failing tests:**
  - `test_train_labels.py`: `test_plantvillage_corn_folders_map` (the four names above give `common_rust`, `northern_leaf_blight`, `gray_leaf_spot`, `healthy`); `test_non_corn_folder_returns_none` (e.g. `Tomato___Late_blight`).
  - `test_finetuned.py`: `test_finetuned_uses_fake_pipeline_like_hf` (inject `FakePipeline`, same shape as Task 6's, assert the mapped `Result`).
- [ ] **Step 2: Run `pytest tests/test_train_labels.py tests/test_finetuned.py -v`; expect FAIL.**
- [ ] **Step 3: Implement** `labels.py` and `finetuned.py`, then write `train.py`. `train.py` is not unit tested; it is verified in the next step.
- [ ] **Step 4: Run `pytest -v`; expect PASS.**
- [ ] **Step 5: Verify training manually.** Download a corn subset of a public dataset (e.g. PlantVillage) into an image-folder layout, then run `python -m train.train --data <dir> --out models/corn --epochs 1`. Expected: `models/corn/config.json` exists, and its `id2label` values are all in `LABELS`. Set `detector: finetuned` in a scratch copy of the config and run 3 sample images through the running app.
- [ ] **Step 6: Commit** `feat: fine-tuned detector and training script`.

---

### Task 8: Wire-up, README and end-to-end demo check

**Files:**
- Create: `README.md`, `tests/test_wiring.py`

**Interfaces:**
- Consumes: everything above.

- [ ] **Step 1: Write failing tests** in `tests/test_wiring.py` for `app.main.build_app` (Task 3): `test_build_app_wires_config_to_detector` (a temp config plus a monkeypatched `app.main.build_detector` returning a fake; POST an image to the returned app and get the fake's result, with `min_confidence` from the config applied); `test_startup_fails_clearly_on_missing_key` (a temp `llm_vision` config with the env var unset raises `ConfigError` whose message contains `ANTHROPIC_API_KEY`).
- [ ] **Step 2: Run `pytest tests/test_wiring.py -v`; expect PASS or FAIL.** If `build_app` from Task 3 already satisfies both tests, they pass immediately and Step 3 is only a fix-up; otherwise fix `build_app` until they pass.
- [ ] **Step 3: Confirm** `uvicorn app.main:build_app --factory` starts the server with the default config.
- [ ] **Step 4: Write `README.md`:** what it is; install (core vs `requirements-ml.txt`); how to set `ANTHROPIC_API_KEY`/`HF_TOKEN` (env var, or `.env`); how to switch backends in `config.yaml`; how to run (`uvicorn app.main:build_app --factory --reload`); how to train; how to add a fourth detector; the Task 6 model choice and observed accuracy; a short two-minute demo script for the judges.
- [ ] **Step 5: Run `pytest -v`; expect all PASS.**
- [ ] **Step 6: End-to-end demo check** for each backend you have working (`llm_vision` at minimum): start the server, upload a healthy leaf, a diseased leaf, a non-corn photo, and a corrupt file. Expected: sensible results, a "doesn't look like a corn leaf" state for the non-corn photo, and a readable error for the corrupt file. Also confirm that starting with the API key unset exits with a message naming `ANTHROPIC_API_KEY`.
- [ ] **Step 7: Commit** `feat: wire app from config, add README`.
