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


def test_leaf_filling_most_of_the_frame_is_never_inverted():
    img = Image.new("RGB", (200, 160), (128, 128, 128))
    ImageDraw.Draw(img).rectangle([12, 0, 199, 159], fill=(40, 140, 50))      # leaf touches 3 sides
    m = leaf_mask(img)
    assert m is not None and m.mean() > 0.8 and not m[80, 2]


def test_frame_full_of_leaf_gets_no_mask():
    img = Image.new("RGB", (200, 160), (40, 140, 50))
    ImageDraw.Draw(img).ellipse([60, 50, 140, 110], fill=(150, 130, 90))     # a tan lesion, no background
    assert leaf_mask(img) is None


def test_leaf_running_the_full_height_of_white_paper_is_masked():
    img = Image.new("RGB", (120, 200), (245, 245, 240))
    ImageDraw.Draw(img).rectangle([35, 0, 85, 199], fill=(70, 130, 40))       # strip touches top and bottom
    m = leaf_mask(img)
    assert m is not None and m[100, 60] and not m[100, 5]


def test_textured_grey_lesion_at_the_leaf_edge_stays_leaf():
    rng = np.random.default_rng(5)
    img = Image.new("RGB", (200, 160), (128, 128, 128))
    ImageDraw.Draw(img).ellipse([30, 30, 170, 130], fill=(40, 140, 50))
    a = np.asarray(img).copy()
    lesion = (slice(60, 100), slice(140, 175))                       # straddles the leaf's right edge
    a[lesion] = np.clip(125 + rng.normal(0, 18, a[lesion].shape), 0, 255).astype(np.uint8)   # grey, speckled
    m = leaf_mask(Image.fromarray(a))
    assert m is not None
    h, w = m.shape
    ys, xs = slice(int(65 * h / 160), int(95 * h / 160)), slice(int(145 * w / 200), int(165 * w / 200))
    assert m[ys, xs].mean() > 0.8                                    # lesion kept as leaf


def test_fragmentation_is_low_for_clean_masks_and_high_for_speckled_ones():
    from data_prep.background import MAX_FRAGMENTATION, fragmentation

    clean = leaf_mask(_lab_leaf())
    speckled = clean.copy()
    rng = np.random.default_rng(0)
    bg = ~speckled
    speckled[bg & (rng.random(speckled.shape) < 0.4)] = True        # leaf pixels scattered over the background
    assert fragmentation(clean) < MAX_FRAGMENTATION < fragmentation(speckled)


def test_mask_edge_is_shrunk_so_no_backdrop_rim_remains():
    img = Image.new("RGB", (200, 160), (0, 0, 0))
    ImageDraw.Draw(img).rectangle([40, 30, 160, 130], fill=(40, 140, 50))
    m = leaf_mask(img)
    h, w = m.shape
    a = np.asarray(img.resize((w, h), Image.BILINEAR), dtype=float)
    assert a[m].min(axis=0)[1] > 60                                   # no dark (blended) pixels inside the mask
