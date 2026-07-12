# Детекция автотранспорта на аэрофото

Выпускной проект курса DLS (Detection), продуктовый трек. Автор курса: Асмус Тимофей, инженер-исследователь ФГАУ ЦИТ.

Задача: детекция автотранспорта и участников дорожного движения (пешеходы, люди, велосипеды, машины, фургоны, грузовики, рикши, рикши с навесом, автобусы, мотоциклы) на кадрах, снятых с дронов. Датасет — [VisDrone2019-DET](https://www.kaggle.com/datasets/kushagrapandya/visdrone-dataset) (AISKYEYE, Tianjin University), 10 классов, 10209 изображений, готовое разбиение train/val/test-dev.

## Демо

Ссылка на приложение на Streamlit Cloud: `TODO — добавить после деплоя`
Видео-презентация: `TODO — добавить ссылку`

## Архитектура

Проект разделён на два независимых сервиса:

- `backend/` — FastAPI-приложение с REST API для инференса (`/predict`, `/models`, `/health`).
- `frontend/` — Streamlit-приложение: загрузка фото, выбор модели, отображение результата.

Локально оба сервиса общаются по HTTP (см. `docker-compose.yml`). На Streamlit Community Cloud нельзя поднять два процесса, поэтому там frontend напрямую импортирует `backend/inference.py` как Python-модуль (embedded-режим, без HTTP) — переключение делается переменной окружения `BACKEND_MODE` (`http` по умолчанию, `embedded` на Cloud). Логика инференса при этом одна и та же в обоих режимах.

```
power-line-defect-detection/
├── backend/            # FastAPI: main.py (роуты), inference.py (общая логика инференса)
├── frontend/           # Streamlit: app.py
├── notebooks/          # Colab-ноутбук обучения моделей
├── models/             # .pt веса (не хранятся в git, см. ниже)
├── streamlit_app.py    # entrypoint для Streamlit Cloud
├── docker-compose.yml  # локальный запуск backend + frontend
└── requirements.txt    # зависимости для Streamlit Cloud (корневой entrypoint)
```

## Модели

Обучены три модели одного семейства — YOLOv8n, YOLOv8s, YOLOv8m. В интерфейсе можно переключаться между ними: быстрая и лёгкая или медленная и точная.

| Модель | Назначение |
|---|---|
| YOLOv8n | Быстрый инференс, ниже точность |
| YOLOv8s | Баланс скорости и точности |
| YOLOv8m | Максимальная точность, ниже скорость |

Метрики обучения (mAP@0.5, mAP@0.5:0.95, время обучения) — в `notebooks/train_vehicle_detection.ipynb` и `models/metrics_summary.csv` после обучения.

## Хранение весов моделей

Веса `.pt` не хранятся в git (см. `.gitignore`) — файлы YOLOv8 после дообучения занимают десятки мегабайт, GitHub для этого не предназначен. Вместо этого:

1. После обучения в Colab веса заливаются на [Hugging Face Hub](https://huggingface.co/) (см. последнюю секцию ноутбука).
2. `backend/inference.py` при первом обращении к модели скачивает веса с Hub автоматически, если задана переменная окружения `HF_REPO_ID`.
3. Для локальной разработки можно просто положить `.pt` файлы в `models/` вручную — тогда `HF_REPO_ID` не нужен.

## Обучение моделей

1. Откройте `notebooks/train_vehicle_detection.ipynb` в Google Colab.
2. Создайте Kaggle API-токен на [kaggle.com/settings](https://www.kaggle.com/settings) (API → Create New Token) — файл `kaggle.json` понадобится для скачивания датасета через `kagglehub`.
3. Выполните все ячейки по порядку — ноутбук скачает датасет VisDrone2019-DET, сконвертирует разметку в формат YOLO и обучит все три модели.
4. В конце ноутбука веса загружаются на Hugging Face Hub (нужен токен с правами `write`).

## Локальный запуск

### Вариант 1: Docker Compose (рекомендуется)

```bash
# Положите обученные .pt файлы в models/ (или задайте HF_REPO_ID в docker-compose.yml)
docker compose up --build
```

- Backend: http://localhost:8000/docs
- Frontend: http://localhost:8501

### Вариант 2: вручную, без Docker

```bash
# Терминал 1 — backend
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# Терминал 2 — frontend
cd frontend
pip install -r requirements.txt
BACKEND_MODE=http BACKEND_URL=http://localhost:8000 streamlit run app.py
```

## Деплой на Streamlit Community Cloud

1. Запушьте репозиторий на GitHub (публичный).
2. На [share.streamlit.io](https://share.streamlit.io) создайте новое приложение, укажите репозиторий и главный файл — `streamlit_app.py` (не `frontend/app.py`).
3. В настройках приложения (Advanced settings → Secrets) добавьте:
   ```toml
   HF_REPO_ID = "your-username/vehicle-detection-yolov8"
   ```
4. Приложение автоматически поставит зависимости из корневого `requirements.txt` и при первом запросе скачает веса моделей с Hugging Face Hub.

## Метрики и оценка

Итоговый mAP@0.5 по каждой модели на отложенном test split VisDrone: YOLOv8n — 0.261, YOLOv8s — 0.319, YOLOv8m — 0.360. Полный отчёт с метриками по классам, графиками и анализом результатов — в [`docs/REPORT.md`](docs/REPORT.md). Исходные данные — в `notebooks/train_vehicle_detection.ipynb` (секция 5) и `models/metrics_summary.csv`.

## Чек-лист требований продуктового трека

- [x] Тюнинг модели (обучение YOLOv8 на кастомном датасете)
- [x] Backend (FastAPI, REST API)
- [x] Frontend (Streamlit)
- [x] Выбор из ≥2 моделей (быстрая/точная) — реализовано 3
- [x] Репозиторий на GitHub
- [ ] Видео-презентация — добавить ссылку после записи
- [ ] Деплой на Streamlit Cloud — добавить ссылку после деплоя
