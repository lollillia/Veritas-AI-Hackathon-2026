# Class folders of corn-only datasets have no plant prefix (e.g. the Kaggle
# "Corn or Maize Leaf Disease Dataset": Blight, Common_Rust, Gray_Leaf_Spot, Healthy).
# In that dataset "Blight" is northern corn leaf blight.
_BARE_CLASSES = {
    "healthy": "healthy",
    "common_rust": "common_rust",
    "gray_leaf_spot": "gray_leaf_spot",
    "blight": "northern_leaf_blight",
    "northern_leaf_blight": "northern_leaf_blight",
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
    if "northern" in name:
        return "northern_leaf_blight"
    if "gray" in name or "grey" in name or "cercospora" in name:
        return "gray_leaf_spot"
    return None
