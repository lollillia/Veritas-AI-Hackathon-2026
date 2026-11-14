def folder_to_label(folder_name: str) -> str | None:
    """Map a public-dataset class folder (PlantVillage style) to one of our labels.

    Returns None for folders that are not corn/maize or not a class we support.
    """
    name = folder_name.lower()
    if "corn" not in name and "maize" not in name:
        return None
    if "healthy" in name:
        return "healthy"
    if "rust" in name:
        return "common_rust"
    if "northern" in name:
        return "northern_leaf_blight"
    if "gray" in name or "grey" in name or "cercospora" in name:
        return "gray_leaf_spot"
    return None
