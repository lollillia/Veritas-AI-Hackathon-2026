import random

import numpy as np
import pytest
from PIL import Image, ImageDraw

from data_prep.background import BACKGROUND_KINDS, leaf_mask, replace_background


def _lab_leaf(bg=(128, 128, 128), size=(200, 160)):
    """A green 'leaf' on a plain background, with a lesion inside that has the background colour."""
    img = Image.new("RGB", size, bg)
    d = ImageDraw.Draw(img)
    d.ellipse([30, 30, 170, 130], fill=(40, 140, 50))
    d.ellipse([90, 70, 110, 90], fill=bg)                     # lesion the same colour as the background
    return img


def test_mask_covers_the_leaf_including_lesions_that_match_the_background():
    m = leaf_mask(_lab_leaf())
    assert m is not None
    h, w = m.shape
    assert m[h // 2, w // 2]                                   # lesion centre counts as leaf
    assert not m[2, 2] and not m[-3, -3]                       # corners are background
    assert 0.3 < m.mean() < 0.7


@pytest.mark.parametrize("bg", [(0, 0, 0), (250, 250, 250), (128, 128, 128)])
def test_mask_works_on_black_white_and_grey_backgrounds(bg):
    assert leaf_mask(_lab_leaf(bg)) is not None


def test_cluttered_field_like_image_gets_no_mask():
    rng = np.random.default_rng(0)
    noisy = Image.fromarray(rng.integers(0, 255, (160, 200, 3), dtype=np.uint8))
    assert leaf_mask(noisy) is None


def test_image_without_a_leaf_gets_no_mask():
    assert leaf_mask(Image.new("RGB", (200, 160), (128, 128, 128))) is None


def test_replace_background_keeps_the_leaf_and_changes_the_background():
    img = _lab_leaf()
    m = leaf_mask(img)
    out = replace_background(img, m, random.Random(1), kind="flat")
    a, b = np.asarray(img, dtype=int), np.asarray(out, dtype=int)
    assert out.size == img.size
    assert np.abs(a[80, 60] - b[80, 60]).max() <= 3            # inside the leaf: unchanged
    assert np.abs(a[:8, :8] - b[:8, :8]).mean() > 5             # background region changed


@pytest.mark.parametrize("kind", BACKGROUND_KINDS)
def test_every_background_kind_works_and_is_deterministic(kind, tmp_path):
    field = tmp_path / "field.jpg"
    Image.fromarray(np.random.default_rng(3).integers(0, 255, (300, 300, 3), dtype=np.uint8)).save(field)
    img = _lab_leaf(); m = leaf_mask(img)
    one = replace_background(img, m, random.Random(7), kind=kind, pool=[field])
    two = replace_background(img, m, random.Random(7), kind=kind, pool=[field])
    assert np.array_equal(np.asarray(one), np.asarray(two))


def test_field_kind_without_a_pool_falls_back_to_another_kind():
    img = _lab_leaf(); m = leaf_mask(img)
    assert replace_background(img, m, random.Random(0), kind="field", pool=[]).size == img.size
