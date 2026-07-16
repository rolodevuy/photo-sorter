"""Configuración: las carpetas que elige el usuario.

Modelo mental: ORIGEN (bandeja de entrada, fotos sin clasificar) -> BIBLIOTECA
(las personas ya ordenadas, una subcarpeta por persona).

La biblioteca es UNA sola: ahí escriben el organizador y el buscador, y de ahí
leen la galería, el import, el corrector y el separador. Antes había dos
ajustes distintos para lo mismo ("output_dir" del organizador y "known_dir" de
las galerías), que podían apuntar a lugares diferentes y hacían que las fotos
organizadas no aparecieran en la galería. Ahora es la clave "library"; las
viejas se siguen leyendo para no romper configuraciones existentes.
"""

import json

from app import CONFIG_PATH, DATA_DIR


def load_config():
    """Devuelve la configuración guardada (claves pueden faltar)."""
    if CONFIG_PATH.is_file():
        with open(CONFIG_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_config(cfg):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


def get_library(cfg=None):
    """Carpeta raíz de la biblioteca de personas (<biblioteca>/<nombre>/).

    Retrocompatible: si todavía no se guardó "library", usa la carpeta de las
    galerías ("known_dir") y, si tampoco, el viejo destino del organizador
    ("output_dir")."""
    cfg = load_config() if cfg is None else cfg
    return cfg.get("library") or cfg.get("known_dir") or cfg.get("output_dir") or ""


def set_library(cfg, folder):
    """Fija la biblioteca y sincroniza las claves viejas, para que cualquier
    código o dato que todavía las mire siga apuntando al mismo lugar."""
    cfg["library"] = folder
    cfg["known_dir"] = folder
    cfg["output_dir"] = folder
    return cfg
