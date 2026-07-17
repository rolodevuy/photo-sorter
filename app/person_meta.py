"""Fechas por persona: última vez que se corrigió su entrenamiento y última
vez que se la buscó en una carpeta. Para saber de un vistazo si ya trabajaste
un perfil o si hace mucho que no lo tocás.

data/person_meta.json = {nombre: {"corrected": "2026-07-13 10:30",
                                   "searched": "2026-07-13 11:00"}}
"""

import json
from datetime import datetime

from app import PERSON_META_PATH


def _load():
    if not PERSON_META_PATH.is_file():
        return {}
    try:
        with open(PERSON_META_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save(data):
    PERSON_META_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PERSON_META_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _now():
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def stamp(name, event):
    """Marca ahora el evento ("corrected" | "searched") de una persona."""
    data = _load()
    data.setdefault(name, {})[event] = _now()
    _save(data)


def stamp_many(names, event):
    """Igual que stamp pero para varias personas (ej. búsqueda de todos)."""
    if not names:
        return
    data = _load()
    ts = _now()
    for n in names:
        data.setdefault(n, {})[event] = ts
    _save(data)


def get(name):
    """{"corrected": str|None, "searched": str|None, "voice": str} de una persona."""
    d = _load().get(name, {})
    return {"corrected": d.get("corrected"), "searched": d.get("searched"),
            "voice": d.get("voice", "")}


def get_voice(name):
    """Cómo habla esta persona: se le suma al prompt del pie de foto.

    Como sabemos de quién es la foto (está en su carpeta), el pie lo escribe ella
    en primera persona. Esto es lo que la hace sonar distinta de las demás: una
    cruda y directa, otra tímida. Vacío = la voz genérica de vlm.SYSTEM_PIE."""
    return (_load().get(name, {}) or {}).get("voice", "")


def set_voice(name, text):
    data = _load()
    text = (text or "").strip()
    if text:
        data.setdefault(name, {})["voice"] = text
    elif name in data:
        data[name].pop("voice", None)
    _save(data)
    return text


def rename(old, new):
    """La persona cambió de nombre: su meta (y su voz) la siguen."""
    data = _load()
    if old in data:
        data[new] = data.pop(old)
        _save(data)


def forget(name):
    data = _load()
    if data.pop(name, None) is not None:
        _save(data)


def humanize(ts):
    """'hace 3 días', 'recién', 'hace 1 año'… a partir de un timestamp. None
    si no hay fecha (nunca)."""
    if not ts:
        return None
    try:
        then = datetime.strptime(ts, "%Y-%m-%d %H:%M")
    except ValueError:
        return ts
    secs = (datetime.now() - then).total_seconds()
    if secs < 90:
        return "recién"
    mins = secs / 60
    if mins < 60:
        return f"hace {int(mins)} min"
    hours = mins / 60
    if hours < 24:
        return f"hace {int(hours)} h"
    days = hours / 24
    if days < 30:
        n = int(days)
        return "ayer" if n == 1 else f"hace {n} días"
    months = days / 30
    if months < 12:
        n = int(months)
        return "hace 1 mes" if n == 1 else f"hace {n} meses"
    years = days / 365
    n = int(years)
    return "hace 1 año" if n == 1 else f"hace {n} años"
