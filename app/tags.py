"""Etiquetas manuales (las escribe el usuario, no las adivina ninguna IA).

Dos niveles:
- Por PERSONA: atributos que valen para toda la persona (ej. Alejandro =
  hombre, pelo largo, rubio). Se heredan a todas sus fotos.
- Por FOTO: palabras propias de una foto puntual (ej. "playa"), además de las
  que hereda de su persona.

data/tags.json = {"people": {nombre: [tags]},
                  "photos": {"<nombre>/<rel>": [tags]}}

Las etiquetas se normalizan (minúsculas, sin espacios de más) para que "Rubio"
y "rubio " cuenten como la misma.
"""

import json

from app import TAGS_PATH


# Atajos iniciales del etiquetador (editables desde la web). Se usan la primera
# vez; después mandan los que el usuario guarde.
DEFAULT_SHORTCUTS = [
    "penetración", "oral", "anal", "vaginal", "doble", "masturbación", "grupal", "solo",
    "de perfil", "de espaldas", "de cuatro", "acostada", "arriba", "parada",
]


def _norm(tag):
    return " ".join(tag.strip().lower().split())


def _clean_list(tags):
    """Normaliza, saca vacías y duplicados, conserva el orden."""
    out = []
    for t in tags:
        n = _norm(t)
        if n and n not in out:
            out.append(n)
    return out


def _load():
    if not TAGS_PATH.is_file():
        return {"people": {}, "photos": {}}
    try:
        with open(TAGS_PATH, encoding="utf-8") as f:
            data = json.load(f)
        data.setdefault("people", {})
        data.setdefault("photos", {})
        return data
    except Exception:
        return {"people": {}, "photos": {}}


def _save(data):
    TAGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    # limpiar entradas vacías para no acumular basura
    data["people"] = {k: v for k, v in data["people"].items() if v}
    data["photos"] = {k: v for k, v in data["photos"].items() if v}
    with open(TAGS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_person(name):
    return list(_load()["people"].get(name, []))


def set_person(name, tags):
    data = _load()
    tags = _clean_list(tags)
    if tags:
        data["people"][name] = tags
    else:
        data["people"].pop(name, None)
    _save(data)
    return tags


def photo_key(name, rel):
    return f"{name}/{rel}"


def get_photo(name, rel):
    return list(_load()["photos"].get(photo_key(name, rel), []))


def set_photo(name, rel, tags):
    data = _load()
    key = photo_key(name, rel)
    tags = _clean_list(tags)
    if tags:
        data["photos"][key] = tags
    else:
        data["photos"].pop(key, None)
    _save(data)
    return tags


def effective_photo(name, rel):
    """Etiquetas que le tocan a una foto: las suyas + las heredadas de su persona."""
    data = _load()
    return _clean_list(data["people"].get(name, []) + data["photos"].get(photo_key(name, rel), []))


def people_tagsets():
    """{nombre: [todas sus etiquetas]} = las de la persona + las de sus fotos.
    Para filtrar la galería por etiqueta ('quién tiene fotos de ese tipo')."""
    data = _load()
    out = {}
    for n, ts in data["people"].items():
        out.setdefault(n, set()).update(ts)
    for key, ts in data["photos"].items():
        if "/" in key:
            out.setdefault(key.split("/", 1)[0], set()).update(ts)
    return {n: sorted(s) for n, s in out.items()}


def all_tags():
    """Todas las etiquetas usadas (para autocompletar), ordenadas."""
    data = _load()
    s = set()
    for v in data["people"].values():
        s.update(v)
    for v in data["photos"].values():
        s.update(v)
    return sorted(s)


def rename_person(old, new):
    """Al renombrar/mover, migra las etiquetas de una persona a otro nombre."""
    data = _load()
    if old in data["people"]:
        data["people"].setdefault(new, [])
        data["people"][new] = _clean_list(data["people"][new] + data["people"].pop(old))
    pref = f"{old}/"
    for key in [k for k in data["photos"] if k.startswith(pref)]:
        data["photos"][f"{new}/{key[len(pref):]}"] = data["photos"].pop(key)
    _save(data)


def forget_person(name):
    """Borra las etiquetas de una persona (y de sus fotos) al olvidarla."""
    data = _load()
    data["people"].pop(name, None)
    pref = f"{name}/"
    for key in [k for k in data["photos"] if k.startswith(pref)]:
        data["photos"].pop(key, None)
    _save(data)


def move_photo_tags(from_name, from_rel, to_name, to_rel):
    """Migra las etiquetas propias de una foto cuando se mueve de persona/carpeta."""
    data = _load()
    src = photo_key(from_name, from_rel)
    if src in data["photos"]:
        data["photos"][photo_key(to_name, to_rel)] = data["photos"].pop(src)
        _save(data)


def get_shortcuts():
    """Atajos del etiquetador: los guardados, o los de fábrica la primera vez."""
    data = _load()
    return list(data.get("shortcuts", DEFAULT_SHORTCUTS))


def set_shortcuts(tags):
    data = _load()
    data["shortcuts"] = _clean_list(tags)
    _save(data)
    return data["shortcuts"]


def search(query_tags):
    """Devuelve (personas, fotos) que tienen TODAS las etiquetas pedidas.

    personas: nombres cuyas etiquetas de persona incluyen todas las buscadas.
    fotos: [(nombre, rel)] cuyas etiquetas PROPIAS incluyen todas las buscadas
    (las que heredan de la persona ya aparecen vía la persona)."""
    q = set(_clean_list(query_tags))
    if not q:
        return [], []
    data = _load()
    people = sorted(n for n, tags in data["people"].items() if q <= set(tags))
    photos = []
    for key, tags in data["photos"].items():
        if q <= set(tags) and "/" in key:
            name, rel = key.split("/", 1)
            photos.append((name, rel))
    photos.sort()
    return people, photos
