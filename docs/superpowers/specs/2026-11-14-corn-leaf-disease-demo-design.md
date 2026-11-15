# Veritas: Corn Leaf Disease Detector — Design

Date: 2026-11-14
Status: Draft, awaiting approval

## Intent and why

A website where a visitor uploads a photo of a corn (maize) leaf and gets back whether the leaf is diseased, and which disease. It is a **demo for a hackathon judging panel**. The priorities are:

1. It works on the first try, live.
2. It looks polished and custom.
3. It is easy to explain in about two minutes.

Accuracy on messy real-world field photos matters less than a smooth demo.

## Goals

- Upload a photo (drag-and-drop or file picker), see a preview, get one result.
- A single result card: label, confidence, one-line explanation.
- A generic **detector framework** supporting three interchangeable backends, selected by a **config file**:
  1. Pretrained corn-disease model (e.g. from Hugging Face).
  2. Our own fine-tuned model (trained on a public corn dataset).
  3. Vision LLM API (e.g. Claude) with a prompt.
- Gracefully reject non-corn-leaf images ("This doesn't look like a corn leaf").
- Adding a fourth backend means adding one file.

## Non-goals

- Not a production or farmer-facing tool: no accounts, no history, no database.
- No side-by-side comparison of models. One result from the configured model.
- No switching models from the UI; only the config file selects the model.
- No mobile-app, offline or camera-capture features.
- No guarantee of accuracy on field photos.

## Anti-goals (failure even if technically working)

- Confidently naming a disease for a selfie or a non-corn leaf.
- Crashing or hanging on a bad file or a missing model/API key during the demo.
- Looking like a stock ML demo template.

## Constraints

- Python (the project already has `main.py`); no Node/build step.
- The project is currently empty (PyCharm scaffold), not a git repo.
- Option 3 needs internet and an API key at demo time; option 2 may need a GPU to train, but not to run.

## Approaches considered

| Approach | Trade-off |
|---|---|
| **FastAPI + one plain HTML/JS page (chosen)** | Full control of the look, plain Python, no build step; a little hand-written front-end code. |
| Gradio / Streamlit | Fastest, but looks generic and is hard to restyle. |
| FastAPI + React | Most polish, but a second toolchain is too much overhead for a hackathon. |

Chosen by the user: FastAPI + plain HTML/JS.

## Design

### Structure

```
config.yaml            # selects the active detector + its settings
app/
  main.py              # FastAPI app: serves page, POST /predict
  config.py            # loads config.yaml
  detectors/
    base.py            # Detector interface + Result type
    registry.py        # name -> detector class
    hf_pretrained.py   # option 1
    finetuned.py       # option 2
    llm_vision.py      # option 3
static/index.html      # upload UI + result card
train/                 # script for option 2 (separate from the web app)
```

### Detector interface

Every detector implements `predict(image_bytes) -> Result`. `Result` has:

- `label`: one of `healthy`, `gray_leaf_spot`, `common_rust`, `northern_leaf_blight`, `northern_leaf_spot`, `not_corn_leaf`
- `confidence`: float 0–1
- `note`: one short sentence for display

All three backends map their output onto this same label set, so the UI and API never care which backend is active.

### Config file

`config.yaml` names the active detector and its settings, for example:

```yaml
detector: llm_vision        # hf_pretrained | finetuned | llm_vision
llm_vision:
  model: claude-sonnet-5-5
  api_key_env: ANTHROPIC_API_KEY
hf_pretrained:
  model_id: <hugging face model id>
finetuned:
  weights_path: models/corn.pt
min_confidence: 0.5         # below this, show "not sure" / not_corn_leaf
```

Loaded once at startup. An invalid name or a missing key/weights fails at startup with a clear message, not mid-demo.

### Secrets (Hugging Face token, API keys)

- Secrets are never stored in `config.yaml` or committed. The config holds only the *name* of the environment variable (e.g. `api_key_env`).
- Hugging Face token: read from the `HF_TOKEN` environment variable (the HF libraries pick it up automatically). `huggingface-cli login`, which stores the token in `~/.cache/huggingface/token`, also works. A token is only needed for private or gated models; public models need none.
- Vision LLM key: `ANTHROPIC_API_KEY` environment variable.
- Optional fallback: a project-local `.env` file, loaded at startup. `.env` must be in `.gitignore` from the start, and a `.env.example` with placeholder names (no real values) is provided.
- A missing required secret fails at startup with a message naming the variable.

### API

`POST /predict` with a multipart image returns the `Result` as JSON. Errors (not an image, too large, backend failure) return a clear message the UI shows in the result card.

### UI

One HTML page: drag-and-drop zone, image preview, spinner while waiting, then a result card with label, confidence bar and note. Non-corn and low-confidence results use a distinct "not sure" state.

### Non-corn / low-confidence handling

- LLM backend: the prompt asks for `not_corn_leaf` explicitly.
- Classifier backends: a confidence below `min_confidence` is shown as "not sure". A true "not a corn leaf" detector for classifiers is a nice-to-have; the simple threshold is the baseline.

## Left to the builder

- Exact HF model choice for option 1 (check license and accuracy on a few sample photos).
- Dataset and base model for option 2 (PlantVillage or similar), and training settings.
- Prompt wording for option 3 and how strictly to parse its output.
- Visual styling, within the "polished, not stock" goal.
- Upload size limit and accepted image types (suggested: JPEG/PNG/WebP, ~10 MB).

## My calls (not discussed with the user)

- YAML for config.
- The fixed label set (five classes plus `not_corn_leaf`; `northern_leaf_spot` was added later at the user's request) and the `min_confidence` threshold approach.
- Training code lives in a separate `train/` folder and is not part of the running web app.
- Fail-fast validation of the config at startup.
- Suggested upload limits above.
- Secrets handling: `HF_TOKEN` env var by default, `.env` as an optional gitignored fallback, plus a `.env.example`.
