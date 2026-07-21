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
    from app import respaldo
    respaldo.respaldar(TAGS_PATH)        # copia la versión buena antes de pisar
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


def person_photo_tag_counts(name):
    """{etiqueta: en cuántas fotos de la persona aparece} — SOLO etiquetas propias
    de foto (no las heredadas de la persona). Para el gestor de etiquetas."""
    data = _load()
    pref = f"{name}/"
    counts = {}
    for key, tags in data["photos"].items():
        if key.startswith(pref):
            for t in tags:
                counts[t] = counts.get(t, 0) + 1
    return counts


def remove_photo_tag(name, tag):
    """Saca `tag` de TODAS las fotos de la persona. Devuelve la lista de fotos
    (rel) que la tenían — sirve para deshacer el borrado."""
    tag = _norm(tag)
    if not tag:
        return []
    data = _load()
    pref = f"{name}/"
    tocadas = []
    for key in list(data["photos"]):
        if key.startswith(pref) and tag in data["photos"][key]:
            data["photos"][key] = [t for t in data["photos"][key] if t != tag]
            tocadas.append(key[len(pref):])
    if tocadas:
        _save(data)
    return tocadas


def add_photo_tag_many(name, rels, tag):
    """Vuelve a poner `tag` en esas fotos (deshacer un borrado masivo)."""
    tag = _norm(tag)
    if not tag:
        return 0
    data = _load()
    n = 0
    for rel in rels:
        key = photo_key(name, rel)
        ts = data["photos"].get(key, [])
        if tag not in ts:
            data["photos"][key] = ts + [tag]
            n += 1
    if n:
        _save(data)
    return n


# --- Favoritas: tu vocabulario. Dos niveles: GLOBAL (toda la app) y por PERSONA.
# Sirven para etiquetar rápido a mano y para pisar la redacción de la IA
# ("tipo de pelo: lacio" -> tu "pelo lacio"). data/tags.json -> {"favs": {...}}
_FAV_GLOBAL = "_global"


def get_favs(name=None):
    """Favoritas: con name=None las globales; con name, las de esa persona."""
    favs = (_load().get("favs") or {})
    return list(favs.get(name or _FAV_GLOBAL, []))


def set_favs(tags, name=None):
    data = _load()
    data.setdefault("favs", {})
    key = name or _FAV_GLOBAL
    tags = _clean_list(tags)
    if tags:
        data["favs"][key] = tags
    else:
        data["favs"].pop(key, None)
    _save(data)
    return tags


def toggle_fav(tag, name=None):
    """Marca/desmarca una etiqueta como favorita. Devuelve la lista resultante."""
    tag = _norm(tag)
    cur = get_favs(name)
    if not tag:
        return cur
    cur = [t for t in cur if t != tag] if tag in cur else cur + [tag]
    return set_favs(cur, name)


def favs_efectivas(name):
    """Las que valen al etiquetar a esta persona: las suyas + las globales."""
    return _clean_list(get_favs(name) + get_favs(None))


def promote_to_person(name, tag):
    """Hace la etiqueta GENERAL: la suma al perfil y la SACA de las fotos sueltas
    (ya la heredan todas). Devuelve las fotos de las que la sacó, para deshacer."""
    tag = _norm(tag)
    if not tag:
        return []
    data = _load()
    propias = data["people"].get(name, [])
    if tag not in propias:
        data["people"][name] = propias + [tag]
    pref = f"{name}/"
    tocadas = []
    for key in list(data["photos"]):
        if key.startswith(pref) and tag in data["photos"][key]:
            data["photos"][key] = [t for t in data["photos"][key] if t != tag]
            tocadas.append(key[len(pref):])
    _save(data)
    return tocadas


def rename_photo_tag(name, old, new):
    """Renombra `old` por `new` en TODAS las fotos de la persona (masivo). Si una
    foto ya tenía `new`, no se duplica. Devuelve en cuántas fotos cambió."""
    old, new = _norm(old), _norm(new)
    if not old or not new or old == new:
        return 0
    data = _load()
    pref = f"{name}/"
    n = 0
    for key in list(data["photos"]):
        if not key.startswith(pref) or old not in data["photos"][key]:
            continue
        nuevos = []
        for t in data["photos"][key]:
            t2 = new if t == old else t
            if t2 not in nuevos:
                nuevos.append(t2)
        data["photos"][key] = nuevos
        n += 1
    if n:
        _save(data)
    return n


def pares_parecidos(umbral=0.82, minimo=4):
    """Pares de etiquetas casi iguales: la misma cosa escrita distinto.

    Devuelve [(a, na, b, nb, parecido)] con la MÁS USADA primero, para que al
    juntarlas se conserve la forma que ya predomina. Sale de un problema real:
    conviven "denuda"/"desnuda", "ojoz azules"/"ojos azules". Cada variante mal
    escrita es un molde que no dispara y un alias que hay que escribir a mano.

    `minimo` evita comparar palabras muy cortas, donde todo se parece a todo."""
    import difflib
    conteo = all_photo_tag_counts()
    vocab = [t for t in conteo if len(t) >= minimo]
    out = []
    for i, a in enumerate(sorted(vocab)):
        for b in sorted(vocab)[i + 1:]:
            if abs(len(a) - len(b)) > 3:
                continue
            r = difflib.SequenceMatcher(None, a, b).ratio()
            if r > umbral:
                x, y = (a, b) if conteo[a] >= conteo[b] else (b, a)
                out.append((x, conteo[x], y, conteo[y], round(r, 2)))
    # primero los que más fotos destraban
    out.sort(key=lambda p: -(p[1] + p[3]))
    return out


def merge_tag_global(vieja, nueva):
    """Reemplaza `vieja` por `nueva` en las fotos de TODAS las personas (y en las
    etiquetas de perfil). Para limpiar el vocabulario de una.

    Devuelve en cuántas fotos cambió. Si una foto ya tenía las dos, no duplica."""
    vieja, nueva = _norm(vieja), _norm(nueva)
    if not vieja or not nueva or vieja == nueva:
        return 0
    data = _load()
    n = 0
    for seccion in ("photos", "people"):
        for clave, lista in list(data[seccion].items()):
            if vieja not in lista:
                continue
            nuevas = []
            for t in lista:
                t2 = nueva if t == vieja else t
                if t2 not in nuevas:
                    nuevas.append(t2)
            data[seccion][clave] = nuevas
            n += 1
    if n:
        _save(data)
    return n


def all_photo_entries():
    """[(clave, [tags])] de cada foto etiquetada. La clave es "<persona>/<rel>",
    así el llamador saca la carpeta (para saber si es _nsfw). Para el generador
    de moldes, que recorre toda la biblioteca."""
    return list(_load()["photos"].items())


def all_photo_tag_counts():
    """{etiqueta: en cuántas fotos aparece} sobre TODA la biblioteca (etiquetas
    de foto, no las de perfil). Para el editor de moldes: saber qué etiquetas son
    frecuentes y cuáles no tienen molde todavía."""
    data = _load()
    counts = {}
    for tags in data["photos"].values():
        for t in set(tags):
            counts[t] = counts.get(t, 0) + 1
    return counts


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
