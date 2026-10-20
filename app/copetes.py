"""Copetes: titulares "de noticia" de cada foto, generados por IA mirando la
imagen (situación, ropa, contexto).

data/copetes.json = { "name/rel": {"copete": "...", "ts": 1730000000.0} }

El set de la home se guarda aparte, SOLO en el navegador (localStorage), no acá.
Este archivo es solo la persistencia en disco de los copetes generados.
"""

import json
import time

from app import COPETES_PATH


def _load():
    if not COPETES_PATH.is_file():
        return {}
    try:
        with open(COPETES_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save(data):
    COPETES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(COPETES_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get(name, rel):
    return _load().get(f"{name}/{rel}")


def get_all():
    return _load()


def set_copete(name, rel, texto):
    """Guarda el copete de una foto. Devuelve el texto guardado."""
    texto = (texto or "").strip()
    data = _load()
    key = f"{name}/{rel}"
    if texto:
        entry = data.get(key) or {}
        entry["copete"] = texto
        entry["ts"] = time.time()
        data[key] = entry
    else:
        data.pop(key, None)
    _save(data)
    return texto


def tener(name, rel):
    """¿Ya hay un copete generado para esta foto?"""
    entry = _load().get(f"{name}/{rel}")
    return bool(entry and (entry.get("copete") or "").strip())


def por_persona(name):
    """Los copetes ya generados de una persona: [(rel, texto), ...]."""
    pref = f"{name}/"
    return [(k[len(pref):], v.get("copete", "")) for k, v in _load().items()
            if k.startswith(pref) and (v.get("copete") or "").strip()]
