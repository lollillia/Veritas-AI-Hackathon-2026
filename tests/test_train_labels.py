import pytest

from train.labels import folder_to_label


@pytest.mark.parametrize(
    "folder,label",
    [
        ("Corn_(maize)___Common_rust_", "common_rust"),
        ("Corn_(maize)___Northern_Leaf_Blight", "northern_leaf_blight"),
        ("Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "gray_leaf_spot"),
        ("Corn_(maize)___healthy", "healthy"),
    ],
)
def test_plantvillage_corn_folders_map(folder, label):
    assert folder_to_label(folder) == label


def test_non_corn_folder_returns_none():
    assert folder_to_label("Tomato___Late_blight") is None


@pytest.mark.parametrize(
    "folder,label",
    [
        ("Healthy", "healthy"),
        ("Common_Rust", "common_rust"),
        ("Gray_Leaf_Spot", "gray_leaf_spot"),
        ("Blight", "northern_leaf_blight"),
    ],
)
def test_bare_class_folders_of_a_corn_only_dataset_map(folder, label):
    assert folder_to_label(folder) == label


@pytest.mark.parametrize("folder", ["Tomato___healthy", "Potato___Late_blight", "Late_blight", "Rice_Brown_spot"])
def test_other_crops_and_unknown_bare_names_are_ignored(folder):
    assert folder_to_label(folder) is None
