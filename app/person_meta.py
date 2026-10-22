"""Fechas por persona: última vez que se corrigió su entrenamiento y última
vez que se la buscó en una carpeta. Para saber de un vistazo si ya trabajaste
un perfil o si hace mucho que no lo tocás.

data/person_meta.json = {nombre: {"corrected": "2026-07-13 10:30",
                                   "searched": "2026-07-13 11:00",
                                   "voice": "…", "bio": "…"}}

Además guarda la voz (cómo habla en los pies de foto) y la bio del perfil,
ambas editables desde el modal de configuración de la persona.
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
    """{"corrected", "searched", "voice", "bio"} de una persona."""
    d = _load().get(name, {})
    return {"corrected": d.get("corrected"), "searched": d.get("searched"),
            "voice": d.get("voice", ""), "bio": d.get("bio", "")}


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


def get_bio(name):
    """La bio del perfil (el párrafo que se ve bajo el nombre). Vacío si no tiene."""
    return (_load().get(name, {}) or {}).get("bio", "")


def get_nombre(name):
    """El nombre "lindo" de la persona (ej. "Nombre Apellido"), para el título del
    álbum y para que la IA la nombre así. Vacío = se usa el nombre de la carpeta."""
    return (_load().get(name, {}) or {}).get("nombre", "")


def get_alias(name):
    """Los alias con que se la puede llamar (lista). Vacío = solo el nombre."""
    al = (_load().get(name, {}) or {}).get("alias") or []
    return [a for a in al if isinstance(a, str) and a.strip()]


def get_identidad(name):
    """{"nombre", "alias"} de una persona: el nombre para mostrar y sus alias."""
    return {"nombre": get_nombre(name), "alias": get_alias(name)}


def set_identidad(name, nombre="", alias=()):
    """Guarda el nombre y los alias de una persona. Vacíos se borran (se cae al
    nombre de la carpeta). `alias` puede ser lista o texto separado por comas."""
    data = _load()
    d = data.setdefault(name, {})
    nombre = (nombre or "").strip()
    if isinstance(alias, str):
        alias = alias.split(",")
    limpios, vistos = [], set()
    for a in alias or ():
        a = (a or "").strip()
        k = a.lower()
        if a and k not in vistos:
            vistos.add(k)
            limpios.append(a)
    if nombre:
        d["nombre"] = nombre
    else:
        d.pop("nombre", None)
    if limpios:
        d["alias"] = limpios
    else:
        d.pop("alias", None)
    _save(data)
    return nombre, limpios


def nombre_para_texto(name):
    """Cómo nombrarla en un texto de la IA: el nombre lindo si lo hay, si no la
    carpeta. Es el ancla; los alias son las variantes (ver variantes_nombre)."""
    return get_nombre(name) or name


def variantes_nombre(name):
    """Las formas de nombrarla que la IA puede usar: la principal primero (nombre
    o carpeta) y después los alias, sin repetir."""
    principal = nombre_para_texto(name)
    out, vistos = [], set()
    for n in [principal] + get_alias(name):
        k = (n or "").lower()
        if n and k not in vistos:
            vistos.add(k)
            out.append(n)
    return out


def set_profile(name, bio="", voice=""):
    """Guarda juntas la bio y la voz (el modal de configuración las edita de una
    sola vez, una sola escritura). Vacías se borran. Devuelve (bio, voice)."""
    data = _load()
    d = data.setdefault(name, {})
    bio, voice = (bio or "").strip(), (voice or "").strip()
    if bio:
        d["bio"] = bio
    else:
        d.pop("bio", None)
    if voice:
        d["voice"] = voice
    else:
        d.pop("voice", None)
    _save(data)
    return bio, voice


def set_cover_src(name, rel, left, top, side):
    """Recuerda de qué foto y con qué recorte salió la portada, para que el editor
    abra directo en esa foto (y en ese recorte) en vez de pedir elegir de nuevo."""
    data = _load()
    data.setdefault(name, {})["cover"] = {"rel": rel, "left": int(left),
                                          "top": int(top), "side": int(side)}
    _save(data)


def get_cover_src(name):
    """{"rel","left","top","side"} de la portada, o None si no se sabe (portadas
    viejas, o puestas antes de que se guardara el origen)."""
    c = (_load().get(name, {}) or {}).get("cover")
    return c if isinstance(c, dict) and c.get("rel") else None


def rename(old, new):
    """La persona cambió de nombre: su meta (voz, bio, fechas) la sigue."""
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
