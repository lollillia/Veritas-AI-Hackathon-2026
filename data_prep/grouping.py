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


def group_ids(hashes: list[int], threshold: int = 4) -> list[int]:
    """Group hashes whose Hamming distance is <= threshold (connected components).

    Returns one id per input, numbered from 0 in order of first appearance.
    """
    n = len(hashes)
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
            if c > i:
                ri, rc = find(i), find(c)
                if ri != rc:
                    parent[max(ri, rc)] = min(ri, rc)

    ids: dict[int, int] = {}
    return [ids.setdefault(find(i), len(ids)) for i in range(n)]
