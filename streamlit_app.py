"""Entrypoint для Streamlit Community Cloud.

Streamlit Cloud запускает один файл из корня репозитория и не поднимает
отдельный backend-процесс, поэтому здесь принудительно включается
embedded-режим (frontend/app.py вызывает backend/inference.py напрямую,
без HTTP) перед импортом основного приложения.
"""
import os
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

os.environ.setdefault("BACKEND_MODE", "embedded")
# frontend/app.py resolves the backend path via Path(__file__) when run standalone,
# but __file__ is not set inside this exec() call — add it here instead, where a
# real __file__ is available.
sys.path.insert(0, os.path.join(ROOT_DIR, "backend"))

with open(os.path.join(ROOT_DIR, "frontend", "app.py"), encoding="utf-8") as f:
    exec(compile(f.read(), "frontend/app.py", "exec"))
