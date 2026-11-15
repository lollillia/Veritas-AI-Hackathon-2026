import pytest

from train.labels import folder_to_label


@pytest.mark.parametrize(
    "folder,label",
    [
        # PlantVillage-style names that mention corn/maize
        ("Corn_(maize)___Common_rust_", "common_rust"),
        ("Corn_(maize)___Northern_Leaf_Blight", "leaf_blight"),
        ("Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "leaf_spot"),
        ("Corn_(maize)___Northern_Leaf_Spot", "leaf_spot"),
        ("Corn_(maize)___healthy", "healthy"),
        # Bare class folders of corn-only datasets (Kaggle corn set, CD&S, RL-NST)
        ("Healthy", "healthy"),
        ("Common_Rust", "common_rust"),
        ("Common_rust", "common_rust"),
        ("Blight", "leaf_blight"),
        ("nlb", "leaf_blight"),
        ("Leaf_Blight", "leaf_blight"),
        ("Northern_Leaf_Blight", "leaf_blight"),
        ("Gray_Leaf_Spot", "leaf_spot"),
        ("gls", "leaf_spot"),
        ("nls", "leaf_spot"),
        ("Leaf_spot", "leaf_spot"),
        ("Northern_Leaf_Spot", "leaf_spot"),
        # Multi-format maize dataset (RAW_Image folders)
        ("Healthy_Leaf", "healthy"),
        ("Corn_Leaf_Blight", "leaf_blight"),
        ("Eyespot", "leaf_spot"),            # user decision: eyespot counts as a leaf spot
        # Kaggle ndisan/corn-leaf-disease (Indonesian folder names)
        ("Daun Sehat", "healthy"),
        ("Hawar Daun", "leaf_blight"),
        ("Karat Daun", "common_rust"),
    ],
)
def test_folder_names_map_to_general_labels(folder, label):
    assert folder_to_label(folder) == label


@pytest.mark.parametrize(
    "folder",
    ["Tomato___healthy", "Tomato___Late_blight", "Potato___Late_blight", "Late_blight", "Rice_Brown_spot", "Streak_virus",
     "Maize_Streak_Disease", "Bacterial_Leaf_Streak",
     "Bercak Daun"],   # copies and rotations of the Kaggle corn set: never used
)
def test_other_crops_unknown_names_and_streak_virus_are_ignored(folder):
    assert folder_to_label(folder) is None
