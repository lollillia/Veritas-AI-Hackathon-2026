"""Standard photo augmentation for training (PIL only, deterministic for a given random.Random).

Field photos differ from lab photos in framing, orientation, light and focus, so each training load gets:
a random crop keeping 60-100% of the area (aspect 3:4 to 4:3), random flips and quarter turns (a leaf has
no fixed orientation), brightness / contrast / saturation changes of up to 25%, and a light blur on 20%.
"""
import math
import random

from PIL import Image, ImageEnhance, ImageFilter

_TURNS = [None, Image.Transpose.ROTATE_90, Image.Transpose.ROTATE_180, Image.Transpose.ROTATE_270]


def standard_augment(image: Image.Image, rng: random.Random) -> Image.Image:
    w, h = image.size
    area = rng.uniform(0.6, 1.0) * w * h
    ratio = math.exp(rng.uniform(math.log(3 / 4), math.log(4 / 3)))
    cw = max(1, min(w, round(math.sqrt(area * ratio))))
    ch = max(1, min(h, round(math.sqrt(area / ratio))))
    x, y = rng.randint(0, w - cw), rng.randint(0, h - ch)
    out = image.convert("RGB").crop((x, y, x + cw, y + ch))
    if rng.random() < 0.5:
        out = out.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    turn = rng.choice(_TURNS)
    if turn is not None:
        out = out.transpose(turn)
    for enhancer in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
        out = enhancer(out).enhance(rng.uniform(0.75, 1.25))
    if rng.random() < 0.2:
        out = out.filter(ImageFilter.GaussianBlur(rng.uniform(0.5, 1.5)))
    return out
