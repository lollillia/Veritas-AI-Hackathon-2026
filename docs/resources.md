# Resources

Papers, code and services this project builds on or compares against. Datasets themselves are listed in
[`data/README.md`](../data/README.md) and [`data-sources.md`](data-sources.md).

## Related work

| Resource | What it is | Why it matters here |
|---|---|---|
| [Sam-B-Y/CV-Corn-Disease-Detection](https://github.com/Sam-B-Y/CV-Corn-Disease-Detection) (GitHub) | PyTorch corn-leaf disease classifier with a paper (Duke University undergraduates, June 2024). Uses the same Kaggle "Corn or Maize Leaf Disease" dataset as our `kaggle` source (3 diseases + healthy) | Found the same problem we did: models overfit to **class-dependent backgrounds**. Fix: OpenCV colour segmentation (RGB → HSV masks) to remove backgrounds before training; compared ResNet, VGG16, EfficientNet, MobileNet, DenseNet, LeNet and ShuffleNet (best after segmentation: DenseNet 0.891, MobileNet 0.877); used LIME to inspect decisions. License not stated on the page. Close to our leaf-mask approach (`data_prep/background.py`), which instead replaces backgrounds during training only |
| Craze, Pillay, Joubert & Berger (2022), "Deep Learning Diagnostics of Gray Leaf Spot in Maize under Mixed Disease Field Conditions", *Plants* 11, 1942. https://doi.org/10.3390/plants11151942 | Paper behind the University of Pretoria field dataset (`data/raw/20237613`) | Shows the PlantVillage background shortcut: 94.1% on PlantVillage vs 55.1% on field photos; background removal with Mask R-CNN did not help on field photos |
| Mohanty, Hughes & Salathé (2016), "Using Deep Learning for Image-Based Plant Disease Detection", *Frontiers in Plant Science* 7, 1419. https://doi.org/10.3389/fpls.2016.01419 | The PlantVillage benchmark result (99.35% on 54,306 images, 38 classes) | Source of most of our lab-style images; see `models/PLANTVILLAGE_BACKGROUND_ISSUE.md` (local) for the problem |
| Ahmad, Saraswat, El Gamal & Johal (2021), "CD&S Dataset: Handheld Imagery Dataset Acquired Under Field Conditions for Corn Disease Identification and Severity Estimation". https://arxiv.org/abs/2110.12084 | Paper behind the CD&S field dataset (`cds`) | Field photos; our only northern leaf spot source |

## Services used by the data pipeline

| Service | Used for | Notes |
|---|---|---|
| [Open-Meteo historical weather archive](https://open-meteo.com) (ERA5 reanalysis) | `data/weather.csv` | free, no key; modelled values on a ~10–25 km grid |
| [OpenStreetMap Nominatim](https://nominatim.openstreetmap.org) | city names in the image metadata (`data/image_metadata.json`), site coordinates | at most one request per second; only rounded coordinates or place names are sent |
| [Kaggle CLI](https://github.com/Kaggle/kaggle-api) | downloading Kaggle datasets | needs `KAGGLE_API_TOKEN` (in `.env`, gitignored) |
