# app/backend/core/conditions.py

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import io


def apply_condition(image: Image.Image, condition: str) -> Image.Image:
    """
    Apply selected distortion to the image.
    """

    if condition == "clean":
        return image

    if condition == "noise":
        return add_noise(image)

    if condition == "blur":
        return image.filter(ImageFilter.GaussianBlur(radius=2))

    if condition == "low_light":
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(0.4)

    if condition == "jpeg":
        return apply_jpeg_compression(image)

    raise ValueError(f"Unknown condition: {condition}")


def add_noise(image: Image.Image) -> Image.Image:
    """
    Add Gaussian noise.
    """
    arr = np.array(image).astype(np.float32)

    noise = np.random.normal(0, 25, arr.shape)
    arr += noise

    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def apply_jpeg_compression(image: Image.Image) -> Image.Image:
    """
    Simulate JPEG compression.
    """
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=30)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")