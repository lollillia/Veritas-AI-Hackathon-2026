import io

import pytest
from PIL import Image

from app.imaging import load_rgb_image, to_jpeg_bytes


def _encode(img: Image.Image, fmt: str, **kw) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt, **kw)
    return buf.getvalue()


def test_rgba_png_becomes_rgb():
    data = _encode(Image.new("RGBA", (8, 8), (10, 200, 30, 128)), "PNG")
    assert load_rgb_image(data).mode == "RGB"


def test_cmyk_jpeg_becomes_rgb():
    data = _encode(Image.new("CMYK", (8, 8), (0, 0, 0, 0)), "JPEG")
    assert load_rgb_image(data).mode == "RGB"


def test_exif_rotation_applied():
    img = Image.new("RGB", (20, 10), (0, 128, 0))
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 CW when displayed
    data = _encode(img, "JPEG", exif=exif)
    assert load_rgb_image(data).size == (10, 20)


def test_garbage_bytes_raise_value_error():
    with pytest.raises(ValueError, match="not a valid image"):
        load_rgb_image(b"not an image")


def test_to_jpeg_bytes_downsizes():
    out = to_jpeg_bytes(Image.new("RGB", (4000, 2000), (0, 128, 0)))
    assert out.startswith(b"\xff\xd8")
    assert max(Image.open(io.BytesIO(out)).size) == 1568
