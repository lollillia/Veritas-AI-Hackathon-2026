import io

from PIL import Image, ImageOps, UnidentifiedImageError


def load_rgb_image(image_bytes: bytes) -> Image.Image:
    """Decode bytes, apply EXIF orientation, convert to RGB."""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.load()
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise ValueError("not a valid image") from exc
    fmt = img.format
    img = ImageOps.exif_transpose(img).convert("RGB")
    img.format = fmt  # callers (the API) check the original format
    return img


def to_jpeg_bytes(image: Image.Image, max_side: int = 1568) -> bytes:
    image = image.convert("RGB")
    if max(image.size) > max_side:
        image = image.copy()
        image.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=90)
    return buf.getvalue()
