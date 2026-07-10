"""FastAPI backend для детекции автотранспорта на аэрофото."""
from __future__ import annotations

import io

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

from inference import ModelNotFoundError, list_available_models, run_inference

app = FastAPI(
    title="Vehicle Detection API",
    description="Детекция автотранспорта на аэрофото (пешеходы, велосипеды, машины, грузовики и др.) с помощью YOLOv8",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class DetectionOut(BaseModel):
    class_name: str
    confidence: float
    box: tuple[float, float, float, float]


class PredictResponse(BaseModel):
    model_id: str
    detections: list[DetectionOut]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/models")
def get_models() -> dict:
    return list_available_models()


@app.post("/predict", response_model=PredictResponse)
async def predict(
    file: UploadFile = File(...),
    model_id: str = Query("yolov8n", description="yolov8n | yolov8s | yolov8m"),
    conf: float = Query(0.25, ge=0.0, le=1.0),
) -> PredictResponse:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Файл должен быть изображением")

    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Не удалось прочитать изображение: {exc}") from exc

    try:
        detections = run_inference(image, model_id, conf=conf)
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return PredictResponse(
        model_id=model_id,
        detections=[
            DetectionOut(class_name=d.class_name, confidence=d.confidence, box=d.box)
            for d in detections
        ],
    )
