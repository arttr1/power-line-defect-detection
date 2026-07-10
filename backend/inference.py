"""Общая логика инференса YOLOv8, переиспользуемая FastAPI-роутами и Streamlit fallback-режимом."""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from PIL import Image
from ultralytics import YOLO

MODELS_DIR = Path(os.environ.get("MODELS_DIR", Path(__file__).resolve().parent.parent / "models"))

# Репозиторий на Hugging Face Hub, куда после обучения в Colab заливаются веса .pt.
# Если переменная не задана, скачивание с Hub отключено и используются только локальные файлы в MODELS_DIR.
HF_REPO_ID = os.environ.get("HF_REPO_ID", "")

AVAILABLE_MODELS = {
    "yolov8n": {
        "weights": "yolov8n_vehicle_detection.pt",
        "label": "YOLOv8n (быстрая, менее точная)",
        "description": "Наименьшая модель, самая высокая скорость инференса, ниже mAP",
    },
    "yolov8s": {
        "weights": "yolov8s_vehicle_detection.pt",
        "label": "YOLOv8s (баланс скорости и точности)",
        "description": "Компромисс между скоростью и качеством детекции",
    },
    "yolov8m": {
        "weights": "yolov8m_vehicle_detection.pt",
        "label": "YOLOv8m (медленная, точная)",
        "description": "Наибольшая модель из набора, максимальная точность, ниже скорость",
    },
}


def _ensure_weights_downloaded(weights_path: Path, filename: str) -> None:
    if weights_path.exists() or not HF_REPO_ID:
        return
    from huggingface_hub import hf_hub_download

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    downloaded = hf_hub_download(repo_id=HF_REPO_ID, filename=filename)
    Path(downloaded).replace(weights_path)


@dataclass
class Detection:
    class_name: str
    confidence: float
    box: tuple[float, float, float, float]  # x1, y1, x2, y2


class ModelNotFoundError(Exception):
    pass


@lru_cache(maxsize=len(AVAILABLE_MODELS))
def load_model(model_id: str) -> YOLO:
    if model_id not in AVAILABLE_MODELS:
        raise ModelNotFoundError(f"Неизвестная модель: {model_id}")
    filename = AVAILABLE_MODELS[model_id]["weights"]
    weights_path = MODELS_DIR / filename

    _ensure_weights_downloaded(weights_path, filename)

    if not weights_path.exists():
        raise ModelNotFoundError(
            f"Файл весов не найден: {weights_path}. "
            "Обучите модели в notebooks/train_vehicle_detection.ipynb, залейте .pt файлы в HuggingFace Hub "
            "(и задайте HF_REPO_ID) либо поместите их в models/ вручную."
        )
    return YOLO(str(weights_path))


def run_inference(image: Image.Image, model_id: str, conf: float = 0.25) -> list[Detection]:
    model = load_model(model_id)
    results = model.predict(image, conf=conf, verbose=False)
    detections: list[Detection] = []
    for result in results:
        names = result.names
        for box in result.boxes:
            cls_id = int(box.cls.item())
            confidence = float(box.conf.item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(
                Detection(
                    class_name=names[cls_id],
                    confidence=confidence,
                    box=(x1, y1, x2, y2),
                )
            )
    return detections


def list_available_models() -> dict:
    return {
        model_id: {
            "label": meta["label"],
            "description": meta["description"],
            "ready": (MODELS_DIR / meta["weights"]).exists() or bool(HF_REPO_ID),
        }
        for model_id, meta in AVAILABLE_MODELS.items()
    }
