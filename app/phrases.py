"""Frases del interrogator reescritas por el usuario.

BLIP describe con frases formularias y repetidas ("una mujer acostada en una
cama" sale en cientos de fotos), y el traductor las deja en español neutro y
medio robótico. En vez de meter un LLM que las reescriba (pesado, y con dialecto
sale caricaturesco), el usuario las reescribe UNA vez y valen para siempre:

- Cada frase automática que aparece se guarda con un contador de cuántas veces
  salió, así se reescriben primero las que más rinden.
- Si la frase tiene versión propia, se muestra esa. Si no, la automática tal cual.

Cada frase lleva además una marca nsfw, para poder editar esa lista aparte (usan
otro vocabulario). Se marca sola si la foto venía de una carpeta _nsfw, y se
puede corregir a mano.

data/phrases.json = {frase_automatica: {"custom": "tu version", "count": 12,
                                        "nsfw": false}}
La clave es la frase automática normalizada (minúsculas, espacios colapsados).
"""

import json

from app import PHRASES_PATH


def _norm(text):
    return " ".join((text or "").strip().lower().split())


def _load():
    if not PHRASES_PATH.is_file():
        return {}
    try:
        with open(PHRASES_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save(data):
    PHRASES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PHRASES_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def apply(auto_text, nsfw=False):
    """Devuelve la frase a mostrar: la reescrita por el usuario si existe, si no
    la automática. Además la registra y le suma al contador.

    nsfw=True (la foto venía de una carpeta _nsfw) marca la frase como NSFW.
    Una vez marcada queda marcada, salvo que la corrijas a mano."""
    key = _norm(auto_text)
    if not key:
        return auto_text
    data = _load()
    entry = data.setdefault(key, {"custom": "", "count": 0, "nsfw": False})
    entry["count"] = entry.get("count", 0) + 1
    entry.setdefault("custom", "")
    if nsfw:
        entry["nsfw"] = True
    entry.setdefault("nsfw", False)
    _save(data)
    return entry["custom"] or auto_text


def set_nsfw(auto_text, value):
    """Marca/desmarca a mano una frase como NSFW."""
    data = _load()
    entry = data.get(_norm(auto_text))
    if entry is None:
        return False
    entry["nsfw"] = bool(value)
    _save(data)
    return entry["nsfw"]


def set_custom(auto_text, custom):
    """Guarda (o borra, si custom viene vacío) tu versión de una frase."""
    key = _norm(auto_text)
    if not key:
        return ""
    data = _load()
    entry = data.setdefault(key, {"custom": "", "count": 0})
    entry["custom"] = (custom or "").strip()
    _save(data)
    return entry["custom"]


def get_custom(auto_text):
    return _load().get(_norm(auto_text), {}).get("custom", "")


def list_all(only_pending=False, nsfw=None):
    """[{auto, custom, count, nsfw}] ordenado por las que más aparecieron.
    only_pending: solo las que todavía no reescribiste.
    nsfw: True = solo NSFW, False = solo las demás, None = todas."""
    out = [{"auto": k, "custom": v.get("custom", ""), "count": v.get("count", 0),
            "nsfw": bool(v.get("nsfw", False))}
           for k, v in _load().items()]
    if only_pending:
        out = [p for p in out if not p["custom"]]
    if nsfw is not None:
        out = [p for p in out if p["nsfw"] is bool(nsfw)]
    out.sort(key=lambda p: (-p["count"], p["auto"]))
    return out


def stats(nsfw=None):
    """Totales; con nsfw=True/False cuenta solo ese subconjunto."""
    items = list_all(nsfw=nsfw)
    hechas = sum(1 for p in items if p["custom"])
    data = _load()
    return {"total": len(items), "hechas": hechas, "pendientes": len(items) - hechas,
            "nsfw_total": sum(1 for v in data.values() if v.get("nsfw"))}


def forget(auto_text):
    """Saca una frase del listado (por si se coló basura)."""
    data = _load()
    if data.pop(_norm(auto_text), None) is not None:
        _save(data)
