"""Qué fotos ya se pasaron por la IA para etiquetar.

Por qué existe: el ✓ del Probador miraba `tags.get_photo(...)`, o sea "tiene
alguna etiqueta". Eso no es lo mismo que "ya la procesé":

- una foto etiquetada A MANO quedaba marcada como hecha y, con "ocultar ya
  etiquetadas", desaparecía sin haber pasado nunca por el modelo;
- una foto que SÍ pasó pero el modelo devolvió vacío quedaba sin marca, así que
  la volvías a mandar y gastabas CPU de nuevo.

Acá se anota el hecho: se procesó tal foto, tal día, y cuántas etiquetas trajo
(0 es un dato válido y útil: es lo que permite no repetirla, o buscar a propósito
las que salieron vacías).

data/procesadas.json = {"<persona>/<rel>": {"fecha": "2026-07-21 14:03", "n": 12}}
"""

import json
from datetime import datetime

from app import DATA_DIR

PROCESADAS_PATH = DATA_DIR / "procesadas.json"


def _load():
    if not PROCESADAS_PATH.is_file():
        return {}
    try:
        with open(PROCESADAS_PATH, encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def _save(d):
    PROCESADAS_PATH.parent.mkdir(parents=True, exist_ok=True)
    from app import respaldo
    respaldo.respaldar(PROCESADAS_PATH)     # misma red que tags/config
    with open(PROCESADAS_PATH, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)


def _key(name, rel):
    return f"{name}/{rel}"


def marcar(name, rel, n_etiquetas):
    """Anota que esta foto ya pasó por la IA. `n_etiquetas` puede ser 0: ese es
    justamente el caso que antes se perdía."""
    d = _load()
    d[_key(name, rel)] = {"fecha": datetime.now().strftime("%Y-%m-%d %H:%M"),
                          "n": int(n_etiquetas)}
    _save(d)


def esta(name, rel):
    return _key(name, rel) in _load()


def de_persona(name):
    """{rel: {"fecha","n"}} de las fotos ya procesadas de esa persona."""
    pref = f"{name}/"
    return {k[len(pref):]: v for k, v in _load().items() if k.startswith(pref)}


def olvidar(name, rel=None):
    """Saca la marca: de una foto, o de TODA la persona si rel=None. Sirve para
    reprocesar a propósito (por ejemplo, las que salieron vacías)."""
    d = _load()
    if rel is not None:
        quitar = [_key(name, rel)]
    else:
        pref = f"{name}/"
        quitar = [k for k in d if k.startswith(pref)]
    for k in quitar:
        d.pop(k, None)
    if quitar:
        _save(d)
    return len(quitar)
