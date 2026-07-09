"""Configuración: carpetas de origen y destino elegidas por el usuario."""

import json

from app import CONFIG_PATH, DATA_DIR


def load_config():
    """Devuelve {"photos_dir": ..., "output_dir": ...} (strings, pueden faltar)."""
    if CONFIG_PATH.is_file():
        with open(CONFIG_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_config(cfg):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
