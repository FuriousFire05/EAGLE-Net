# app/backend/core/security.py

from fastapi import UploadFile, HTTPException
from PIL import Image
import io

ALLOWED_TYPES = ["image/jpeg", "image/png", "image/webp"]
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB


async def validate_image(file: UploadFile):
    """
    Validate uploaded image securely.
    """

    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    contents = await file.read()

    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large")

    try:
        image = Image.open(io.BytesIO(contents))
        image = image.convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    return image