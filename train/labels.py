# Class folders of corn-only datasets have no plant prefix (e.g. the Kaggle corn dataset:
# Blight, Common_Rust, Gray_Leaf_Spot, Healthy; CD&S: gls / nlb / nls; RL-NST: Leaf_Blight,
# Leaf_spot, ...). All blights are one label and all leaf spots are one label.
# RL-NST's Streak_virus folder is deliberately not mapped, so it is never trained on.
_BARE_CLASSES = {
    "healthy": "healthy",
    "common_rust": "common_rust",
    "blight": "leaf_blight",
    "leaf_blight": "leaf_blight",
    "northern_leaf_blight": "leaf_blight",
    "nlb": "leaf_blight",
    "leaf_spot": "leaf_spot",
    "gray_leaf_spot": "leaf_spot",
    "northern_leaf_spot": "leaf_spot",
    "gls": "leaf_spot",
    "nls": "leaf_spot",
}


def folder_to_label(folder_name: str) -> str | None:
    """Map a dataset class folder to one of our labels.

    Handles PlantVillage-style names that mention corn/maize (e.g.
    "Corn_(maize)___Common_rust_") and the bare class names of corn-only datasets.
    Returns None for other crops and for classes we do not support.
    """
    name = folder_name.lower()
    if "corn" not in name and "maize" not in name:
        return _BARE_CLASSES.get(name.replace(" ", "_").replace("-", "_"))
    if "healthy" in name:
        return "healthy"
    if "rust" in name:
        return "common_rust"
    if "blight" in name:
        return "leaf_blight"
    if "spot" in name or "cercospora" in name:
        return "leaf_spot"
    return None
