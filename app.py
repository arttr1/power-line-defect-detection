"""Entrypoint для Hugging Face Spaces (Gradio SDK)."""
from frontend.app_gradio import demo

if __name__ == "__main__":
    demo.launch()
