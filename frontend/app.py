"""Streamlit-приложение для детекции автотранспорта на аэрофото.

Работает в двух режимах, переключаемых переменной окружения BACKEND_MODE:
- "http" (по умолчанию локально) — ходит в отдельный FastAPI backend по BACKEND_URL.
- "embedded" (используется на Streamlit Cloud, где нельзя поднять второй процесс) —
  вызывает backend/inference.py напрямую как Python-модуль, без HTTP.
"""
from __future__ import annotations

import io
import os
import sys
import time
from pathlib import Path

import requests
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

BACKEND_MODE = os.environ.get("BACKEND_MODE", "http")
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

if BACKEND_MODE == "embedded":
    # When this file is executed directly (`streamlit run frontend/app.py`), __file__
    # is set and we can resolve the backend path ourselves. When it's exec()'d from
    # streamlit_app.py (Streamlit Cloud / Render entrypoint), __file__ is not defined
    # here — that entrypoint is responsible for putting backend/ on sys.path instead.
    try:
        backend_dir = str(Path(__file__).resolve().parent.parent / "backend")
        if backend_dir not in sys.path:
            sys.path.insert(0, backend_dir)
    except NameError:
        pass
    from inference import ModelNotFoundError, list_available_models, run_inference  # noqa: E402

CLASS_COLORS = {
    "pedestrian": "#e74c3c",
    "people": "#e67e22",
    "bicycle": "#f39c12",
    "car": "#2ecc71",
    "van": "#1abc9c",
    "truck": "#3498db",
    "tricycle": "#9b59b6",
    "awning-tricycle": "#8e44ad",
    "bus": "#c0392b",
    "motor": "#16a085",
}


def get_models() -> dict:
    if BACKEND_MODE == "embedded":
        return list_available_models()
    response = requests.get(f"{BACKEND_URL}/models", timeout=10)
    response.raise_for_status()
    return response.json()


def predict(image_bytes: bytes, model_id: str, conf: float) -> list[dict]:
    if BACKEND_MODE == "embedded":
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        try:
            detections = run_inference(image, model_id, conf=conf)
        except ModelNotFoundError as exc:
            raise RuntimeError(str(exc)) from exc
        return [
            {"class_name": d.class_name, "confidence": d.confidence, "box": d.box}
            for d in detections
        ]

    files = {"file": ("image.jpg", image_bytes, "image/jpeg")}
    params = {"model_id": model_id, "conf": conf}
    response = requests.post(f"{BACKEND_URL}/predict", files=files, params=params, timeout=60)
    response.raise_for_status()
    return response.json()["detections"]


def draw_detections(image: Image.Image, detections: list[dict]) -> Image.Image:
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    try:
        font = ImageFont.truetype("Arial.ttf", 16)
    except OSError:
        font = ImageFont.load_default()

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        color = CLASS_COLORS.get(det["class_name"], "#ffffff")
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = f"{det['class_name']} {det['confidence']:.2f}"
        text_bbox = draw.textbbox((x1, y1), label, font=font)
        draw.rectangle(
            [text_bbox[0], text_bbox[1] - 2, text_bbox[2] + 4, text_bbox[3] + 2],
            fill=color,
        )
        draw.text((x1 + 2, y1 - 2), label, fill="white", font=font)
    return annotated


st.set_page_config(page_title="Детекция автотранспорта", page_icon="🚗", layout="wide")
st.title("🚗 Детекция автотранспорта на аэрофото")
st.caption(
    "Выпускной проект DLS. Детекция автотранспорта и участников дорожного движения "
    "(пешеходы, велосипеды, машины, грузовики, автобусы и др.) на кадрах с дронов "
    "на базе YOLOv8."
)

try:
    models_info = get_models()
except Exception as exc:
    st.error(f"Не удалось получить список моделей: {exc}")
    st.stop()

with st.sidebar:
    st.header("Настройки")
    model_options = list(models_info.keys())
    model_labels = {k: v["label"] for k, v in models_info.items()}
    selected_model = st.selectbox(
        "Модель детекции",
        options=model_options,
        format_func=lambda k: model_labels.get(k, k),
    )
    st.caption(models_info[selected_model]["description"])
    if not models_info[selected_model]["ready"]:
        st.warning("Веса этой модели не найдены. Сначала обучите модель (см. notebooks/).")

    conf_threshold = st.slider("Порог уверенности (confidence)", 0.05, 0.95, 0.25, 0.05)

uploaded_file = st.file_uploader("Загрузите аэрофото", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_bytes = uploaded_file.getvalue()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Исходное изображение")
        st.image(image, use_container_width=True)

    if st.button("Запустить детекцию", type="primary"):
        with st.spinner("Выполняется детекция..."):
            start = time.time()
            try:
                detections = predict(image_bytes, selected_model, conf_threshold)
            except Exception as exc:
                st.error(f"Ошибка при детекции: {exc}")
                st.stop()
            elapsed = time.time() - start

        with col2:
            st.subheader("Результат детекции")
            annotated = draw_detections(image, detections)
            st.image(annotated, use_container_width=True)

        st.success(f"Найдено объектов: {len(detections)}. Время инференса: {elapsed:.2f} с")

        if detections:
            st.subheader("Детали обнаружений")
            st.dataframe(
                [
                    {
                        "Класс": d["class_name"],
                        "Confidence": round(d["confidence"], 3),
                        "BBox (x1, y1, x2, y2)": tuple(round(v, 1) for v in d["box"]),
                    }
                    for d in detections
                ],
                use_container_width=True,
            )
