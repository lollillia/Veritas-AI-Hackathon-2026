import random

import numpy as np
from PIL import Image

from train.augment import standard_augment


def _img(w=120, h=80):
    arr = np.random.default_rng(0).integers(0, 255, (h, w, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def test_returns_an_rgb_crop_no_larger_than_the_input():
    img = _img()
    for seed in range(30):
        out = standard_augment(img, random.Random(seed))
        assert out.mode == "RGB"
        area = out.size[0] * out.size[1]
        assert 0.3 * 120 * 80 <= area <= 120 * 80          # a crop that keeps most of the leaf


def test_same_seed_gives_the_same_image():
    img = _img()
    a = standard_augment(img, random.Random(7))
    b = standard_augment(img, random.Random(7))
    assert a.size == b.size and np.array_equal(np.asarray(a), np.asarray(b))


def test_different_seeds_give_different_images():
    img = _img()
    outs = {np.asarray(standard_augment(img, random.Random(s))).tobytes() for s in range(10)}
    assert len(outs) == 10


def test_input_image_is_not_modified():
    img = _img()
    before = np.asarray(img).copy()
    standard_augment(img, random.Random(1))
    assert np.array_equal(np.asarray(img), before)
