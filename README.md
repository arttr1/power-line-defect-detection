# Детекция автотранспорта на аэрофото

Выпускной проект курса DLS (Detection), продуктовый трек. Автор курса: Асмус Тимофей, инженер-исследователь ФГАУ ЦИТ.

Задача: детекция автотранспорта и участников дорожного движения (пешеходы, люди, велосипеды, машины, фургоны, грузовики, рикши, рикши с навесом, автобусы, мотоциклы) на кадрах, снятых с дронов. Датасет — [VisDrone2019-DET](https://www.kaggle.com/datasets/kushagrapandya/visdrone-dataset) (AISKYEYE, Tianjin University), 10 классов, 10209 изображений, готовое разбиение train/val/test-dev.

## Демо

Приложение: https://vehicle-detection-4uid.onrender.com
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
├── models/             # .pt веса моделей (хранятся в git, см. ниже)
├── streamlit_app.py    # entrypoint для деплоя (embedded-режим)
├── docker-compose.yml  # локальный запуск backend + frontend
└── requirements.txt    # зависимости для деплоя (корневой entrypoint)
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

Веса `.pt` (суммарно ~81 МБ на все три модели) хранятся прямо в git, в `models/`. Это осознанный компромисс: платформы с эфемерной файловой системой на бесплатном тарифе (например, Render Free) не гарантируют, что файл, скачанный во время выполнения запроса, переживёт следующий рестарт процесса — скачивание "по требованию" с Hugging Face Hub при каждом холодном старте оказалось ненадёжным. Хранение весов в репозитории убирает эту точку отказа.

1. После обучения в Colab веса опционально заливаются на [Hugging Face Hub](https://huggingface.co/) (см. последнюю секцию ноутбука) — как резервная копия и для скачивания вне репозитория.
2. `backend/inference.py` в первую очередь ищет веса локально в `models/`. Если файла нет и задана переменная окружения `HF_REPO_ID`, делается попытка скачать его с Hub — это резервный путь для сред с постоянной файловой системой.
3. После обучения новых моделей замените `.pt` файлы в `models/` и закоммитьте изменения.

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

## Деплой на Render

Streamlit Community Cloud (share.streamlit.io) на момент написания недоступен по сети из региона разработки, поэтому приложение развёрнуто на [Render](https://render.com) как обычный Web Service:

1. Запушьте репозиторий на GitHub (публичный).
2. На [dashboard.render.com](https://dashboard.render.com) создайте New → Web Service, укажите публичный URL репозитория.
3. Настройки сборки:
   - Build command: `pip install -r requirements.txt`
   - Start command: `streamlit run streamlit_app.py --server.port $PORT --server.address 0.0.0.0`
4. Веса моделей уже в репозитории (`models/`), поэтому `HF_REPO_ID` задавать не обязательно — но можно оставить как резервный путь.
5. Бесплатный тариф Render (512 МБ RAM) укладывает YOLOv8n/s/m, но инстанс засыпает после 15 минут простоя — первый запрос после паузы будет медленнее.

## Метрики и оценка

Итоговый mAP@0.5 по каждой модели на отложенном test split VisDrone: YOLOv8n — 0.261, YOLOv8s — 0.319, YOLOv8m — 0.360. Полный отчёт с метриками по классам, графиками и анализом результатов — в [`docs/REPORT.md`](docs/REPORT.md). Исходные данные — в `notebooks/train_vehicle_detection.ipynb` (секция 5) и `models/metrics_summary.csv`.

## Чек-лист требований продуктового трека

- [x] Тюнинг модели (обучение YOLOv8 на кастомном датасете)
- [x] Backend (FastAPI, REST API)
- [x] Frontend (Streamlit)
- [x] Выбор из ≥2 моделей (быстрая/точная) — реализовано 3
- [x] Репозиторий на GitHub
- [x] Деплой (Render): https://vehicle-detection-4uid.onrender.com
- [ ] Видео-презентация — добавить ссылку после записи
