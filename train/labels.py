"""Dataset folder name -> label. The study covers fungal leaf diseases only.

Labels: healthy, common_rust, northern_leaf_blight, gray_leaf_spot, northern_leaf_spot, eyespot.
Non-fungal classes (maize streak virus, maize chlorotic mottle virus, bacterial leaf streak) and the
ndisan "Bercak Daun" folder (copies and rotations of the Kaggle corn set) are deliberately unmapped,
so they are never trained on.
"""

# Bare class-folder names of corn-only datasets, normalised to lower case with underscores.
_BARE_CLASSES = {
    # healthy
    "healthy": "healthy", "healthy_leaf": "healthy", "daun_sehat": "healthy",
    # common rust (Puccinia sorghi); "Karat Daun" (leaf rust) is assumed to be common rust
    "common_rust": "common_rust", "karat_daun": "common_rust",
    # northern leaf blight (Exserohilum turcicum); generic "blight" folders are assumed to be it
    "blight": "northern_leaf_blight", "leaf_blight": "northern_leaf_blight",
    "northern_leaf_blight": "northern_leaf_blight", "nlb": "northern_leaf_blight", "hawar_daun": "northern_leaf_blight",
    # gray leaf spot (Cercospora zeae-maydis); RL-NST "Leaf_spot" is gray leaf spot (user-confirmed)
    "gray_leaf_spot": "gray_leaf_spot", "gls": "gray_leaf_spot", "leaf_spot": "gray_leaf_spot",
    # northern leaf spot (Bipolaris zeicola)
    "northern_leaf_spot": "northern_leaf_spot", "nls": "northern_leaf_spot",
    # eyespot (Kabatiella zeae); "northern anthracnose" is merged into it (user decision)
    "eyespot": "eyespot", "northern_anthracnose": "eyespot",
}


def folder_to_label(folder_name: str) -> str | None:
    """Map a dataset class folder to a label, or None for anything not used."""
    name = folder_name.lower().replace(" ", "_").replace("-", "_")
    if "corn" not in name and "maize" not in name:
        return _BARE_CLASSES.get(name)
    if "healthy" in name:
        return "healthy"
    if "rust" in name:
        return "common_rust"
    if "anthracnose" in name or "eyespot" in name:
        return "eyespot"
    if "blight" in name:
        return "northern_leaf_blight"
    if "northern" in name and "spot" in name:
        return "northern_leaf_spot"
    if "gray" in name or "grey" in name or "cercospora" in name:
        return "gray_leaf_spot"
    return None                                            # e.g. maize streak, chlorotic mottle virus
