"""Near-duplicate detection with a 64-bit difference hash."""
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

_M1 = np.uint64(0x5555555555555555)
_M2 = np.uint64(0x3333333333333333)
_M4 = np.uint64(0x0F0F0F0F0F0F0F0F)
_H01 = np.uint64(0x0101010101010101)


def dhash(path: Path | str) -> int:
    """64-bit difference hash: grayscale, 9x8, compare each pixel with its right neighbour."""
    try:
        with Image.open(path) as img:
            gray = img.convert("L").resize((9, 8), Image.LANCZOS)
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise ValueError("not a valid image") from exc
    px = np.asarray(gray, dtype=np.int16)
    bits = (px[:, 1:] > px[:, :-1]).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def _popcount(x: np.ndarray) -> np.ndarray:
    x = x - ((x >> np.uint64(1)) & _M1)
    x = (x & _M2) + ((x >> np.uint64(2)) & _M2)
    x = (x + (x >> np.uint64(4))) & _M4
    return (x * _H01) >> np.uint64(56)


def group_ids(hashes: list[int], threshold: int = 4, keys: list | None = None) -> list[int]:
    """Group hashes whose Hamming distance is <= threshold (connected components).

    With keys (e.g. labels), two hashes are only linked when their keys are equal: smooth,
    low-texture photos of different classes can have close hashes, and chaining would
    otherwise merge them. Returns one id per input, numbered from 0 in order of first appearance.
    """
    n = len(hashes)
    if keys is not None and len(keys) != n:
        raise ValueError("keys must have one entry per hash")
    if n == 0:
        return []
    arr = np.array(hashes, dtype=np.uint64)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    chunk = 512
    for start in range(0, n, chunk):
        block = arr[start:start + chunk]
        dist = _popcount(block[:, None] ^ arr[None, :])
        rows, cols = np.nonzero(dist <= threshold)
        for r, c in zip(rows.tolist(), cols.tolist()):
            i = start + r
            if c > i and (keys is None or keys[i] == keys[c]):
                ri, rc = find(i), find(c)
                if ri != rc:
                    parent[max(ri, rc)] = min(ri, rc)

    ids: dict[int, int] = {}
    return [ids.setdefault(find(i), len(ids)) for i in range(n)]


def _number(parent: list[int]) -> list[int]:
    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    ids: dict[int, int] = {}
    return [ids.setdefault(find(i), len(ids)) for i in range(len(parent))]


def _union(parent: list[int], a: int, b: int) -> None:
    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    ra, rb = find(a), find(b)
    if ra != rb:
        parent[max(ra, rb)] = min(ra, rb)


def burst_group_ids(times: list, keys: list, window_s: float = 20.0) -> list[int]:
    """Group shots taken within window_s seconds of the previous shot with the same key (chained).

    times are datetimes or None (no timestamp: the image stays alone). Used for datasets that
    photograph each leaf several times in a row, so all shots of one leaf stay in one split.
    """
    parent = list(range(len(times)))
    by_key: dict = {}
    for i, (t, k) in enumerate(zip(times, keys)):
        if t is not None:
            by_key.setdefault(k, []).append((t, i))
    for shots in by_key.values():
        shots.sort()
        for (t1, i1), (t2, i2) in zip(shots, shots[1:]):
            if (t2 - t1).total_seconds() <= window_s:
                _union(parent, i1, i2)
    return _number(parent)


def merge_partitions(a: list[int], b: list[int]) -> list[int]:
    """Union of two groupings of the same items: items grouped in either end up together."""
    parent = list(range(len(a)))
    for ids in (a, b):
        first: dict[int, int] = {}
        for i, g in enumerate(ids):
            if g in first:
                _union(parent, first[g], i)
            else:
                first[g] = i
    return _number(parent)
