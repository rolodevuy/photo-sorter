"""Saca de data/ todo lo que ya no tiene respaldo en la biblioteca.

El problema que resuelve: la app guarda etiquetas, pies, comentarios y miniaturas
indexados por NOMBRE DE CARPETA. Si borrás carpetas desde el Explorador (o vaciás
la biblioteca entera), la app nunca se entera y esa metadata queda colgada de
nombres que ya no existen. No rompe nada —se busca por carpeta y no aparece—,
pero ensucia: el autocompletado te sigue ofreciendo el vocabulario de la base
vieja y las miniaturas ocupan disco al pedo.

Corre sola al arrancar (ver flask_app.main), en un hilo aparte para no demorar la
ventana. La idea es que la biblioteca sea la ÚNICA fuente de verdad: si la
vaciaste, se vació.

LA GUARDA QUE IMPORTA: si la carpeta de la biblioteca no está accesible, esto NO
BORRA NADA y se va. La biblioteca vive en un disco aparte (E:), y un disco sin
montar se ve igual que una biblioteca vacía: sin esta guarda, abrir la app con el
disco desconectado te borraría toda la metadata de una.
"""

import hashlib
import json
import shutil
from pathlib import Path

from app import (DATA_DIR, DUPES_PATH, DUPES_THUMBS, GALLERY_THUMBS, KNOWN_PATH,
                 KNOWN_THUMBS, PERSON_META_PATH, POSTS_PATH, TAGS_PATH)


def _leer(path):
    if not Path(path).is_file():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _escribir(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _fotos_reales(raiz, carpetas):
    """{"persona/rel"} de todo lo que HAY en el disco. rel con "/" como separador,
    igual que las claves de tags.json y posts.json."""
    vivas = set()
    for nombre in carpetas:
        base = raiz / nombre
        try:
            for f in base.rglob("*"):
                if f.is_file():
                    vivas.add(f"{nombre}/{f.relative_to(base).as_posix()}")
        except OSError:
            continue          # carpeta ilegible: se la deja tranquila
    return vivas


def purgar(library, dry_run=False):
    """Borra la metadata huérfana. Devuelve un dict con el informe.

    dry_run=True cuenta lo que sacaría sin tocar nada."""
    inf = {"ok": False, "motivo": "", "personas": 0, "fotos": 0, "posts": 0,
           "meta": 0, "entrenadas": 0, "miniaturas": 0, "mb": 0.0}

    raiz = Path(library) if library else None
    if not raiz or not raiz.is_dir():
        inf["motivo"] = "la biblioteca no está accesible (¿disco desconectado?)"
        return inf
    try:
        carpetas = {d.name for d in raiz.iterdir() if d.is_dir()}
    except OSError as e:
        inf["motivo"] = f"no se pudo leer la biblioteca: {e}"
        return inf

    vivas = _fotos_reales(raiz, carpetas)
    inf["ok"] = True

    # --- etiquetas: personas sin carpeta, y fotos que ya no están en el disco
    tags = _leer(TAGS_PATH)
    if isinstance(tags, dict):
        gente = tags.get("people", {})
        fuera = [n for n in gente if n not in carpetas]
        fotos = tags.get("photos", {})
        fuera_f = [k for k in fotos if k not in vivas]
        # las favoritas por persona cuelgan del mismo nombre ("_global" se queda:
        # es tu vocabulario, no depende de que exista ninguna carpeta)
        favs = tags.get("favs", {})
        fuera_v = [n for n in favs if n != "_global" and n not in carpetas]
        if (fuera or fuera_f or fuera_v) and not dry_run:
            for n in fuera:
                gente.pop(n, None)
            for k in fuera_f:
                fotos.pop(k, None)
            for n in fuera_v:
                favs.pop(n, None)
            _escribir(TAGS_PATH, tags)
        inf["personas"], inf["fotos"] = len(fuera), len(fuera_f)

    # --- pies y comentarios (mismas claves "persona/rel")
    posts = _leer(POSTS_PATH)
    if isinstance(posts, dict):
        fuera = [k for k in posts if k not in vivas]
        if fuera and not dry_run:
            for k in fuera:
                posts.pop(k, None)
            _escribir(POSTS_PATH, posts)
        inf["posts"] = len(fuera)

    # --- fechas, voz y portada por persona
    meta = _leer(PERSON_META_PATH)
    if isinstance(meta, dict):
        fuera = [n for n in meta if n not in carpetas]
        if fuera and not dry_run:
            for n in fuera:
                meta.pop(n, None)
            _escribir(PERSON_META_PATH, meta)
        inf["meta"] = len(fuera)

    # --- caras entrenadas (y su miniatura). Es lo más caro de rehacer, pero una
    # persona sin carpeta no se puede usar para nada.
    known = _leer(KNOWN_PATH)
    if isinstance(known, dict):
        fuera = [n for n in known if n not in carpetas]
        if fuera and not dry_run:
            for n in fuera:
                known.pop(n, None)
            _escribir(KNOWN_PATH, known)
        inf["entrenadas"] = len(fuera)

    # --- miniaturas de galería: el nombre es sha1("persona/rel"), así que se sabe
    # exactamente cuáles sobran sin adivinar
    if GALLERY_THUMBS.is_dir():
        validas = {hashlib.sha1(k.encode("utf-8")).hexdigest() + ".jpg" for k in vivas}
        for t in GALLERY_THUMBS.glob("*.jpg"):
            if t.name not in validas:
                inf["miniaturas"] += 1
                inf["mb"] += t.stat().st_size / 1e6
                if not dry_run:
                    t.unlink(missing_ok=True)

    # --- miniaturas de una corrida puntual (duplicados, NSFW, búsquedas…). Son
    # cache de un resultado: si el índice que las explicaba ya no está, sobran
    # todas. Se numeran por posición, no por nombre, así que es todo o nada.
    for carpeta, indice in ((DUPES_THUMBS, DUPES_PATH),
                            (DATA_DIR / "nsfw_thumbs", None),
                            (DATA_DIR / "find_thumbs", None),
                            (DATA_DIR / "pick_thumbs", None),
                            (DATA_DIR / "correct_thumbs", None),
                            (DATA_DIR / "split_thumbs", None)):
        if not carpeta.is_dir():
            continue
        # con índice: sobra solo si el índice desapareció. Sin índice: son de una
        # corrida y se regeneran solas cada vez que la lanzás.
        if indice is not None and Path(indice).is_file():
            continue
        archivos = [f for f in carpeta.iterdir() if f.is_file()]
        if not archivos:
            continue
        inf["miniaturas"] += len(archivos)
        inf["mb"] += sum(f.stat().st_size for f in archivos) / 1e6
        if not dry_run:
            shutil.rmtree(carpeta, ignore_errors=True)

    # la miniatura de cada persona conocida sí va por nombre
    if KNOWN_THUMBS.is_dir():
        from app.known import safe_name
        validos = {safe_name(n) + ".jpg" for n in carpetas}
        for t in KNOWN_THUMBS.glob("*.jpg"):
            if t.name not in validos:
                inf["miniaturas"] += 1
                inf["mb"] += t.stat().st_size / 1e6
                if not dry_run:
                    t.unlink(missing_ok=True)

    inf["mb"] = round(inf["mb"], 1)
    return inf


def resumen(inf):
    """El informe en una línea, para la consola."""
    if not inf["ok"]:
        return f"[limpieza] omitida: {inf['motivo']}"
    partes = [f"{inf['personas']} personas", f"{inf['fotos']} etiquetas de foto",
              f"{inf['posts']} pies", f"{inf['meta']} fichas",
              f"{inf['entrenadas']} caras", f"{inf['miniaturas']} miniaturas"]
    total = (inf["personas"] + inf["fotos"] + inf["posts"] + inf["meta"]
             + inf["entrenadas"] + inf["miniaturas"])
    if not total:
        return "[limpieza] data/ ya estaba en orden"
    return f"[limpieza] se fue lo huérfano: {', '.join(partes)} ({inf['mb']} MB)"
