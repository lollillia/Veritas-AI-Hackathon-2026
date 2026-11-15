"""Background replacement for lab-style leaf photos (training augmentation).

PlantVillage-style photos put each class on its own kind of background, so a model can classify by
background instead of disease. For photos with a plain background, the leaf is cut out (leaf_mask)
and, during training only, pasted onto a random background (replace_background): a flat colour, a
gradient, blotchy noise, or a patch of a field photo from the training split.
"""
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

BACKGROUND_KINDS = ("flat", "gradient", "noise", "field")
LAB_SOURCES = {"kaggle", "rlnst", "ndisan"}             # plain-background photos that get masks
FIELD_SOURCES = {"up_field", "cds", "mendeley_bd", "neau"}  # field photos used as background patches
WORK_SIZE = 160          # masks are computed on a thumbnail this size
BORDER = 4               # border strip (pixels) used to estimate the background colour
MIN_PLAIN_BORDER = 0.6   # share of border pixels that must match the background colour
MIN_COVER, MAX_COVER = 0.08, 0.97


def _otsu(values: np.ndarray) -> float:
    hist, edges = np.histogram(values, bins=64)
    centres = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist); w1 = w0[-1] - w0
    m0 = np.cumsum(hist * centres) / np.maximum(w0, 1)
    m1 = (np.sum(hist * centres) - np.cumsum(hist * centres)) / np.maximum(w1, 1)
    return float(centres[np.argmax(w0 * w1 * (m0 - m1) ** 2)])


def _shift_or(m: np.ndarray) -> np.ndarray:
    g = m.copy()
    g[1:] |= m[:-1]; g[:-1] |= m[1:]; g[:, 1:] |= m[:, :-1]; g[:, :-1] |= m[:, 1:]
    return g


def leaf_mask(img: Image.Image) -> np.ndarray | None:
    """Boolean leaf mask on a WORK_SIZE thumbnail, or None when the background is not plain.

    The background is the background-coloured region connected to the image border, so lesions
    inside the leaf that happen to match the background colour stay part of the leaf.
    """
    small = img.convert("RGB")
    small.thumbnail((WORK_SIZE, WORK_SIZE))
    a = np.asarray(small, dtype=np.float32)
    border = np.concatenate([a[:BORDER].reshape(-1, 3), a[-BORDER:].reshape(-1, 3),
                             a[:, :BORDER].reshape(-1, 3), a[:, -BORDER:].reshape(-1, 3)])
    bg = np.median(border, axis=0)
    if (np.linalg.norm(border - bg, axis=1) < 30).mean() < MIN_PLAIN_BORDER:
        return None                                        # cluttered border: not a lab-style photo
    dist = np.linalg.norm(a - bg, axis=2)
    bg_like = dist < min(max(_otsu(dist), 20.0), 90.0)
    background = np.zeros_like(bg_like)
    background[0, :] = background[-1, :] = background[:, 0] = background[:, -1] = True
    background &= bg_like
    while True:
        grown = _shift_or(background) & bg_like
        if np.array_equal(grown, background):
            break
        background = grown
    leaf = ~background
    leaf = _shift_or(~_shift_or(~leaf))                    # opening: drop single-pixel specks
    if not MIN_COVER <= leaf.mean() <= MAX_COVER:
        return None
    return leaf


def _background(kind: str, size: tuple[int, int], rng: random.Random, pool) -> Image.Image:
    w, h = size
    g = np.random.default_rng(rng.getrandbits(32))
    if kind == "flat":
        return Image.new("RGB", size, tuple(rng.randrange(256) for _ in range(3)))
    if kind == "gradient":
        c1, c2 = g.integers(0, 256, 3), g.integers(0, 256, 3)
        angle = rng.random() * np.pi
        yy, xx = np.mgrid[0:h, 0:w]
        t = (np.cos(angle) * xx / max(w - 1, 1) + np.sin(angle) * yy / max(h - 1, 1))
        t = (t - t.min()) / max(t.max() - t.min(), 1e-6)
        return Image.fromarray((c1 * (1 - t[..., None]) + c2 * t[..., None]).astype(np.uint8))
    if kind == "noise":
        cells = rng.choice((4, 8, 16, 32))
        base = g.integers(0, 256, (cells, cells, 3), dtype=np.uint8)
        return Image.fromarray(base).resize(size, Image.BILINEAR)
    path = rng.choice(list(pool))                           # "field"
    with Image.open(path) as im:
        im.draft("RGB", (max(w, h) * 2, max(w, h) * 2))
        im = im.convert("RGB")
    cw, ch = max(1, int(im.width * rng.uniform(0.3, 0.7))), max(1, int(im.height * rng.uniform(0.3, 0.7)))
    x, y = rng.randrange(im.width - cw + 1), rng.randrange(im.height - ch + 1)
    return im.crop((x, y, x + cw, y + ch)).resize(size, Image.BILINEAR)


def replace_background(img: Image.Image, mask: np.ndarray, rng: random.Random, kind: str | None = None,
                       pool=()) -> Image.Image:
    """The leaf from img (mask from leaf_mask) on a new background; edges are softened."""
    if kind is None:
        kind = rng.choice(BACKGROUND_KINDS)
    if kind == "field" and not pool:
        kind = rng.choice(BACKGROUND_KINDS[:3])
    img = img.convert("RGB")
    alpha = Image.fromarray((mask * 255).astype(np.uint8)).resize(img.size, Image.BILINEAR)
    alpha = alpha.filter(ImageFilter.GaussianBlur(radius=max(1, min(img.size) // 200)))
    return Image.composite(img, _background(kind, img.size, rng, pool), alpha)


def load_mask(cache_dir: Path, md5: str) -> np.ndarray | None:
    p = Path(cache_dir) / f"{md5}.png"
    if not p.is_file():
        return None
    return np.asarray(Image.open(p).convert("L")) > 127
