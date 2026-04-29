# app/backend/routes/compare.py

from fastapi import APIRouter, UploadFile, File
from typing import Literal

from app.backend.core.security import validate_image
from app.backend.core.preprocessing import preprocess_image
from app.backend.core.inference import run_inference
from app.backend.core.conditions import apply_condition

router = APIRouter()

MODEL_LIST = ["baseline_cnn", "lightweight_cnn", "eagle_net"]


@router.post("/compare")
async def compare(
    file: UploadFile = File(...),
    condition: Literal["clean", "noise", "blur", "low_light", "jpeg"] = "clean",
):
    """
    Run inference across all models for comparison.
    """

    image = await validate_image(file)

    # Apply same condition to all models
    image = apply_condition(image, condition)

    tensor = preprocess_image(image)

    results = []

    for model_name in MODEL_LIST:
        result = run_inference(model_name, tensor)
        results.append(result)

    return {
        "condition": condition,
        "results": results
    }