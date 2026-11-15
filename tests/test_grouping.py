from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from data_prep.grouping import dhash, group_ids


def _structured(size=64):
    """A deterministic image with real structure (not a flat colour)."""
    img = Image.new("L", (size, size), 90)
    d = ImageDraw.Draw(img)
    for i, shade in enumerate([200, 40, 160, 70, 220]):
        r = size // 2 - i * size // 12
        d.ellipse([size // 2 - r, size // 2 - r // 2, size // 2 + r, size // 2 + r], fill=shade)
    d.rectangle([2, 2, size // 3, size // 5], fill=250)
    return img.convert("RGB")


def _distance(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def test_identical_files_get_same_hash(tmp_path):
    img = _structured()
    img.save(tmp_path / "a.png")
    img.save(tmp_path / "renamed_copy.png")
    assert dhash(tmp_path / "a.png") == dhash(tmp_path / "renamed_copy.png")


def test_resized_copy_is_within_threshold(tmp_path):
    big = _structured(64)
    big.save(tmp_path / "big.png")
    big.resize((48, 48), Image.LANCZOS).save(tmp_path / "small.jpg", quality=90)
    assert _distance(dhash(tmp_path / "big.png"), dhash(tmp_path / "small.jpg")) <= 4


def test_different_images_are_far_apart(tmp_path):
    left_to_right = Image.linear_gradient("L").rotate(90).convert("RGB")
    right_to_left = left_to_right.transpose(Image.FLIP_LEFT_RIGHT)
    left_to_right.save(tmp_path / "a.png")
    right_to_left.save(tmp_path / "b.png")
    assert _distance(dhash(tmp_path / "a.png"), dhash(tmp_path / "b.png")) > 4


def test_group_ids_links_chains():
    a, b, c, d = 0, 0b1111, 0b1111111, (1 << 64) - 1  # a~b (4 bits), b~c (3 bits), a-c 7 bits
    ids = group_ids([a, b, c, d], threshold=4)
    assert ids[0] == ids[1] == ids[2]
    assert ids[3] != ids[0]


def test_group_ids_are_numbered_by_first_appearance():
    x, y, x_near = 0, (1 << 63) | 0xFFFF, 1
    assert group_ids([x, y, x_near]) == [0, 1, 0]


def test_group_ids_empty_and_single():
    assert group_ids([]) == []
    assert group_ids([12345]) == [0]


def test_group_ids_threshold_is_inclusive():
    assert group_ids([0, 0b1111], threshold=4) == [0, 0]
    assert group_ids([0, 0b11111], threshold=4) == [0, 1]


def test_group_ids_handles_many_hashes_quickly():
    import random
    import time

    rng = random.Random(0)
    hashes = [rng.getrandbits(64) for _ in range(6000)]
    start = time.time()
    ids = group_ids(hashes)
    assert time.time() - start < 20
    assert len(ids) == 6000


def test_dhash_rejects_garbage(tmp_path):
    p = tmp_path / "x.jpg"
    p.write_bytes(b"this is not an image")
    with pytest.raises(ValueError, match="not a valid image"):
        dhash(p)


def test_group_ids_only_links_within_the_same_key():
    # identical hashes but different labels must not be grouped (smooth leaves collide)
    assert group_ids([0, 0, 1], keys=["healthy", "leaf_spot", "healthy"]) == [0, 1, 0]


def test_group_ids_keys_must_match_length():
    with pytest.raises(ValueError):
        group_ids([0, 1], keys=["a"])


def test_burst_groups_chain_shots_within_the_window_per_key():
    from datetime import datetime, timedelta

    from data_prep.grouping import burst_group_ids

    t0 = datetime(2023, 7, 28, 17, 14, 0)
    times = [t0, t0 + timedelta(seconds=12), t0 + timedelta(seconds=30),     # chained: 12 s, then 18 s
             t0 + timedelta(seconds=80),                                      # 50 s later: new burst
             t0 + timedelta(seconds=5),                                       # other folder: separate
             None]                                                            # no timestamp: alone
    keys = ["a", "a", "a", "a", "b", "a"]
    ids = burst_group_ids(times, keys, window_s=20)
    assert ids[0] == ids[1] == ids[2]
    assert len({ids[0], ids[3], ids[4], ids[5]}) == 4


def test_merge_partitions_unions_both_groupings():
    from data_prep.grouping import merge_partitions

    assert merge_partitions([0, 0, 1, 2, 3], [0, 1, 1, 2, 3]) == [0, 0, 0, 1, 2]
    assert merge_partitions([], []) == []


def test_group_ids_allow_callback_can_veto_links():
    allow = lambda i, j: (i, j) != (0, 1)
    assert group_ids([0, 0, 255], allow=allow) == [0, 1, 2]       # the only close pair (0, 1) is vetoed
    assert group_ids([0, 0, 255]) == [0, 0, 1]
