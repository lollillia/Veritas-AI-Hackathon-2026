import pytest

from train.labels import folder_to_label


@pytest.mark.parametrize(
    "folder,label",
    [
        # PlantVillage-style names that mention corn/maize
        ("Corn_(maize)___Common_rust_", "common_rust"),
        ("Corn_(maize)___Northern_Leaf_Blight", "northern_leaf_blight"),
        ("Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "gray_leaf_spot"),
        ("Corn_(maize)___Northern_Leaf_Spot", "northern_leaf_spot"),
        ("Corn_(maize)___healthy", "healthy"),
        # Kaggle corn set (data/)
        ("Healthy", "healthy"), ("Common_Rust", "common_rust"), ("Blight", "northern_leaf_blight"),
        ("Gray_Leaf_Spot", "gray_leaf_spot"),
        # RL-NST (user-confirmed: Leaf_Blight is northern, Leaf_spot is gray)
        ("Common_rust", "common_rust"), ("Leaf_Blight", "northern_leaf_blight"), ("Leaf_spot", "gray_leaf_spot"),
        # CD&S
        ("gls", "gray_leaf_spot"), ("nlb", "northern_leaf_blight"), ("nls", "northern_leaf_spot"),
        # multi-format maize (RAW_Image)
        ("Corn_Leaf_Blight", "northern_leaf_blight"), ("Healthy_Leaf", "healthy"), ("Eyespot", "eyespot"),
        # Kaggle ndisan (Indonesian)
        ("Daun Sehat", "healthy"), ("Hawar Daun", "northern_leaf_blight"), ("Karat Daun", "common_rust"),
        # Mendeley Bangladesh (Final corn dataset)
        ("Gray_leaf_spot", "gray_leaf_spot"),
        # Northeast Agricultural University; anthracnose merged into eyespot (user decision)
        ("healthy maize leaves", "healthy"), ("maize northern leaf blight", "northern_leaf_blight"),
        ("maize northern anthracnose", "eyespot"),
        # University of Pretoria "Diseases of maize in the field" (one-hot CSV columns)
        ("GLS", "gray_leaf_spot"), ("NCLB", "northern_leaf_blight"), ("PLS", "phaeosphaeria_leaf_spot"),
        ("CR", "common_rust"), ("NoFoliarSymptoms", "healthy"), ("Phaeosphaeria_Leaf_Spot", "phaeosphaeria_leaf_spot"),
        ("SR", "southern_rust"),
        # descriptive labels written in image_metadata.csv (china_2, xtu502)
        ("southern_rust", "southern_rust"), ("Southern Rusts", "southern_rust"),
        ("southern_leaf_blight", "southern_leaf_blight"), ("brown_spot", "brown_spot"),
        ("curvularia_leaf_spot", "curvularia_leaf_spot"),
    ],
)
def test_folder_names_map_to_fungal_disease_labels(folder, label):
    assert folder_to_label(folder) == label


@pytest.mark.parametrize(
    "folder",
    ["Tomato___healthy", "Tomato___Late_blight", "Potato___Late_blight", "Late_blight", "Rice_Brown_spot",
     # not fungal: never used
     "Streak_virus", "Maize_Streak_Disease", "Bacterial_Leaf_Streak", "Bacterial Leaf Streak",
     "Maize Chlorotic Mottle Virus",
     # copies and rotations of the Kaggle corn set
     "Bercak Daun",
     # University of Pretoria columns without a label: other, unknown
     "Other", "UnidentifiedDisease"],
)
def test_other_crops_non_fungal_diseases_and_copies_are_ignored(folder):
    assert folder_to_label(folder) is None
