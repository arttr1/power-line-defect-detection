"""Entrypoint для Streamlit Community Cloud.

Streamlit Cloud запускает один файл из корня репозитория и не поднимает
отдельный backend-процесс, поэтому здесь принудительно включается
embedded-режим (frontend/app.py вызывает backend/inference.py напрямую,
без HTTP) перед импортом основного приложения.
"""
import os

os.environ.setdefault("BACKEND_MODE", "embedded")

with open(os.path.join(os.path.dirname(__file__), "frontend", "app.py"), encoding="utf-8") as f:
    exec(compile(f.read(), "frontend/app.py", "exec"))
