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
