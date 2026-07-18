"""Gradio-приложение для детекции автотранспорта на аэрофото.

Используется для деплоя на Hugging Face Spaces (Gradio SDK). Работает в embedded-режиме —
вызывает backend/inference.py напрямую как Python-модуль, без HTTP-сервиса.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
from inference import ModelNotFoundError, list_available_models, run_inference  # noqa: E402

import gradio as gr  # noqa: E402

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


def draw_detections(image: Image.Image, detections: list) -> Image.Image:
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    try:
        font = ImageFont.truetype("Arial.ttf", 16)
    except OSError:
        font = ImageFont.load_default()

    for det in detections:
        x1, y1, x2, y2 = det.box
        color = CLASS_COLORS.get(det.class_name, "#ffffff")
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = f"{det.class_name} {det.confidence:.2f}"
        text_bbox = draw.textbbox((x1, y1), label, font=font)
        draw.rectangle(
            [text_bbox[0], text_bbox[1] - 2, text_bbox[2] + 4, text_bbox[3] + 2],
            fill=color,
        )
        draw.text((x1 + 2, y1 - 2), label, fill="white", font=font)
    return annotated


MODELS_INFO = list_available_models()
MODEL_CHOICES = [(meta["label"], model_id) for model_id, meta in MODELS_INFO.items()]
DEFAULT_MODEL = MODEL_CHOICES[0][1] if MODEL_CHOICES else None


def run_detection(image: Image.Image, model_id: str, conf: float):
    if image is None:
        raise gr.Error("Сначала загрузите изображение")

    start = time.time()
    try:
        detections = run_inference(image.convert("RGB"), model_id, conf=conf)
    except ModelNotFoundError as exc:
        raise gr.Error(str(exc)) from exc
    elapsed = time.time() - start

    annotated = draw_detections(image.convert("RGB"), detections)
    summary = f"Найдено объектов: {len(detections)}. Время инференса: {elapsed:.2f} с"

    table = [
        [d.class_name, round(d.confidence, 3), tuple(round(v, 1) for v in d.box)]
        for d in detections
    ]
    return annotated, summary, table


with gr.Blocks(title="Детекция автотранспорта на аэрофото") as demo:
    gr.Markdown(
        "# 🚗 Детекция автотранспорта на аэрофото\n"
        "Выпускной проект DLS. Детекция автотранспорта и участников дорожного движения "
        "(пешеходы, велосипеды, машины, грузовики, автобусы и др.) на кадрах с дронов "
        "на базе YOLOv8."
    )

    with gr.Row():
        with gr.Column(scale=1):
            model_dropdown = gr.Dropdown(
                choices=MODEL_CHOICES,
                value=DEFAULT_MODEL,
                label="Модель детекции",
            )
            conf_slider = gr.Slider(
                minimum=0.05, maximum=0.95, value=0.25, step=0.05,
                label="Порог уверенности (confidence)",
            )
            input_image = gr.Image(type="pil", label="Загрузите аэрофото")
            run_button = gr.Button("Запустить детекцию", variant="primary")

        with gr.Column(scale=1):
            output_image = gr.Image(type="pil", label="Результат детекции")
            summary_text = gr.Textbox(label="Итог", interactive=False)
            detections_table = gr.Dataframe(
                headers=["Класс", "Confidence", "BBox (x1, y1, x2, y2)"],
                label="Детали обнаружений",
            )

    run_button.click(
        fn=run_detection,
        inputs=[input_image, model_dropdown, conf_slider],
        outputs=[output_image, summary_text, detections_table],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
