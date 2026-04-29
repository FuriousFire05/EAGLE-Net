# app/backend/routes/predict.py

from fastapi import APIRouter, UploadFile, File
from typing import Literal

from app.backend.core.security import validate_image
from app.backend.core.preprocessing import preprocess_image
from app.backend.core.inference import run_inference
from app.backend.core.conditions import apply_condition

router = APIRouter()


@router.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model_name: Literal["baseline_cnn", "lightweight_cnn", "eagle_net"] = "eagle_net",
    condition: Literal["clean", "noise", "blur", "low_light", "jpeg"] = "clean",
):
    """
    Run prediction with optional distortion.
    """

    image = await validate_image(file)

    # APPLY CONDITION
    image = apply_condition(image, condition)

    tensor = preprocess_image(image)

    result = run_inference(model_name, tensor)

    result["condition"] = condition

    return result