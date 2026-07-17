"""Buscador de persona: revisa una carpeta y encuentra en qué fotos y videos
aparece una persona conocida.

Elegís a alguien de la base de conocidos, le pasás una carpeta cualquiera y
recorre todos sus archivos (imágenes, y videos muestreados a 1 fotograma por
segundo) comparando cada cara detectada contra esa persona. El rigor es
ajustable: más estricto = menos falsos positivos, más permisivo = encuentra
ángulos raros a costa de algún error.

Los resultados se muestran en la web con miniaturas y desde ahí se pueden
mover los archivos a la carpeta de la persona (la misma de las galerías).
"""

import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from app import FIND_THUMBS, IMAGE_EXTENSIONS, VIDEO_EXTENSIONS
from app.analyze import BOX_MARGIN, THUMBNAIL_SIZE, _big_enough, _load_rgb, _scan_video
from app.facedet import detect_and_encode
from app.known import load_known

# Rango del deslizador de rigor (distancia coseno). El centro (0.50) es el
# umbral estricto del reconocimiento (known.MATCH_EPS).
EPS_MIN = 0.35
EPS_MAX = 0.65
EPS_DEFAULT = 0.50

# El buscador NO usa el promedio de la persona (si una persona tiene caras
# mezcladas de varias personas, su promedio degenera en una "cara genérica"
# que matchea con cualquiera). Compara contra cada cara guardada real: una
# cara de la carpeta cuenta solo si se parece a un ejemplo concreto.
#
# Además, al buscar a todos, el mejor candidato tiene que ganarle al segundo
# por este margen; si dos conocidos empatan (gemelos, o la misma persona
# cargada con dos nombres), la cara se considera ambigua y no se asigna.
MARGIN = 0.05


def _save_thumb(image, box, out_path):
    """Miniatura del rostro encontrado (recorte con margen, como en analyze)."""
    top, right, bottom, left = box
    h, w = image.shape[:2]
    my = int((bottom - top) * BOX_MARGIN)
    mx = int((right - left) * BOX_MARGIN)
    crop = image[max(0, top - my):min(h, bottom + my),
                 max(0, left - mx):min(w, right + mx)]
    thumb = Image.fromarray(crop)
    thumb.thumbnail((THUMBNAIL_SIZE, THUMBNAIL_SIZE))
    thumb.save(out_path, quality=85)


def _list_media(folder, media="both"):
    """Lista archivos según media: "images" (solo fotos), "videos" (solo
    videos) o "both" (los dos)."""
    exts = {
        "images": IMAGE_EXTENSIONS,
        "videos": VIDEO_EXTENSIONS,
    }.get(media, IMAGE_EXTENSIONS | VIDEO_EXTENSIONS)
    files = [p for p in sorted(folder.rglob("*")) if p.suffix.lower() in exts]
    if not files:
        que = {"images": "imágenes", "videos": "videos"}.get(media, "imágenes ni videos")
        raise ValueError(f"No hay {que} en {folder}")
    return files


def _reset_thumbs():
    if FIND_THUMBS.is_dir():
        shutil.rmtree(FIND_THUMBS, ignore_errors=True)
    FIND_THUMBS.mkdir(parents=True, exist_ok=True)


def _detections(path, should_stop=None):
    """Caras de un archivo (imagen o video): [(box, enc, imagen_rgb), ...]."""
    if path.suffix.lower() in VIDEO_EXTENSIONS:
        kept, _ = _scan_video(path, should_stop)
        return [(k["box"], k["enc"], k["frame"]) for k in kept]
    img = _load_rgb(path)
    return [(box, enc, img)
            for box, enc in detect_and_encode(img) if _big_enough(box)]


def find_known(folder, eps, media="both", progress=None, should_stop=None):
    """Busca a TODOS los conocidos en la carpeta, en una sola pasada.

    Cada cara detectada se compara contra los ejemplos reales de cada persona
    y se asigna al dueño del ejemplo más cercano, solo si le gana al segundo
    candidato por MARGIN (empate = ambigua, no se asigna); así "gonza" no se
    lleva las fotos de "gonzalo". Devuelve
    ([{"name", "hits": [{"rel", "dist", "thumb"}]}] ordenado por cantidad,
    escaneados). Un archivo con dos conocidos aparece en ambos grupos.
    """
    db = load_known()
    if not db:
        raise ValueError("No hay personas conocidas todavía.")
    names = sorted(db, key=str.lower)
    # todos los vectores apilados, con el índice de su dueño (una sola
    # multiplicación por cara, aunque haya miles de ejemplos guardados)
    all_vecs = np.vstack([np.asarray(db[n], dtype=np.float32) for n in names])
    owners = np.concatenate([np.full(len(db[n]), i, dtype=np.int32)
                             for i, n in enumerate(names)])

    folder = Path(folder)
    files = _list_media(folder, media)
    _reset_thumbs()

    by_person = {}   # nombre -> [{"rel", "dist", "thumb"}]
    counter = 0
    scanned = 0
    for i, path in enumerate(files, 1):
        if should_stop and should_stop():
            break
        rel = path.relative_to(folder).as_posix()
        if progress:
            progress(i, len(files), rel)
        scanned += 1
        try:
            dets = _detections(path, should_stop)
        except Exception:
            continue

        # mejor cara del archivo por persona (puede haber varias personas)
        best = {}   # nombre -> (dist, box, imagen, enc)
        for box, enc, image in dets:
            d_all = 1.0 - all_vecs @ enc
            per_person = np.full(len(names), 2.0, dtype=np.float32)
            np.minimum.at(per_person, owners, d_all)   # ejemplo más cercano de cada uno
            order = np.argsort(per_person)
            j, d1 = int(order[0]), float(per_person[order[0]])
            d2 = float(per_person[order[1]]) if len(order) > 1 else 2.0
            # dentro del rigor Y claramente mejor que el segundo candidato
            if d1 <= eps and d2 - d1 >= MARGIN:
                if names[j] not in best or d1 < best[names[j]][0]:
                    best[names[j]] = (d1, box, image, enc)

        for name, (d, box, image, enc) in best.items():
            thumb_name = f"{counter:05d}.jpg"
            counter += 1
            _save_thumb(image, box, FIND_THUMBS / thumb_name)
            by_person.setdefault(name, []).append(
                {"rel": rel, "dist": round(d, 3), "thumb": thumb_name,
                 "enc": enc.tolist()})   # vector de la cara: alimenta la base al mover

    for hits in by_person.values():
        hits.sort(key=lambda h: h["dist"])
    grouped = [{"name": n, "hits": h} for n, h in by_person.items()]
    grouped.sort(key=lambda g: len(g["hits"]), reverse=True)
    return grouped, scanned


def who_is_in(image_path, eps=EPS_DEFAULT, exclude=None):
    """Quiénes de los conocidos están en ESTA foto. Devuelve [nombres], del más
    seguro al menos.

    Misma lógica que find_known (ejemplos reales + MARGIN contra el segundo
    candidato), pero para un archivo suelto. Lo usa el pie de foto para poder
    decir "acá comiendo con Alberto" en vez de "con alguien".

    Barato: es detección de caras (décimas de segundo), no el modelo de lenguaje.
    `exclude` es el dueño de la carpeta, que ya sabemos que está."""
    db = load_known()
    if not db:
        return []
    names = sorted(db, key=str.lower)
    all_vecs = np.vstack([np.asarray(db[n], dtype=np.float32) for n in names])
    owners = np.concatenate([np.full(len(db[n]), i, dtype=np.int32)
                             for i, n in enumerate(names)])
    try:
        dets = _detections(Path(image_path))
    except Exception:
        return []

    best = {}
    for box, enc, image in dets:
        d_all = 1.0 - all_vecs @ enc
        per_person = np.full(len(names), 2.0, dtype=np.float32)
        np.minimum.at(per_person, owners, d_all)
        order = np.argsort(per_person)
        j, d1 = int(order[0]), float(per_person[order[0]])
        d2 = float(per_person[order[1]]) if len(order) > 1 else 2.0
        if d1 <= eps and d2 - d1 >= MARGIN:
            if names[j] not in best or d1 < best[names[j]]:
                best[names[j]] = d1
    for n in (exclude or ()):
        best.pop(n, None)
    return sorted(best, key=lambda n: best[n])


def find_person(folder, name, eps, media="both", progress=None, should_stop=None):
    """Busca la cara de `name` en la carpeta. Devuelve (resultados, escaneados):
    resultados = [{"rel", "dist", "thumb"}] del más parecido al menos, con la
    miniatura del mejor match de cada archivo guardada en data/find_thumbs/.
    Lanza ValueError si la persona no está en la base o la carpeta está vacía.
    """
    db = load_known()
    if name not in db:
        raise ValueError(f"No hay ninguna persona conocida llamada {name}.")
    person_vecs = np.asarray(db[name], dtype=np.float32)

    folder = Path(folder)
    files = _list_media(folder, media)
    _reset_thumbs()

    results = []
    scanned = 0
    for i, path in enumerate(files, 1):
        if should_stop and should_stop():
            break
        rel = path.relative_to(folder).as_posix()
        if progress:
            progress(i, len(files), rel)
        scanned += 1
        try:
            dets = _detections(path, should_stop)
        except Exception:
            continue   # archivo ilegible: se saltea

        # el mejor match del archivo: la cara más parecida a algún ejemplo
        # real de la persona (no a su promedio)
        best = None
        for box, enc, image in dets:
            dist = float((1.0 - person_vecs @ enc).min())
            if dist <= eps and (best is None or dist < best[0]):
                best = (dist, box, image, enc)
        if best:
            thumb_name = f"{len(results):05d}.jpg"
            _save_thumb(best[2], best[1], FIND_THUMBS / thumb_name)
            results.append({"rel": rel, "dist": round(best[0], 3), "thumb": thumb_name,
                            "enc": best[3].tolist()})   # vector: alimenta la base al mover

    results.sort(key=lambda r: r["dist"])
    return results, scanned


def move_to_person(folder, known_dir, name, rels, copy_only=()):
    """Mueve los archivos elegidos a <carpeta_de_galerías>/<nombre>/. Si ya hay
    un archivo con ese nombre, agrega _2, _3... Los que estén en `copy_only`
    se copian en vez de moverse (archivos compartidos con otro conocido: el
    original queda hasta que lo mueva el último). Devuelve (movidos, errores)."""
    folder = Path(folder)
    dest_dir = Path(known_dir) / name
    dest_dir.mkdir(parents=True, exist_ok=True)

    moved, errors = [], []
    for rel in rels:
        src = folder / rel
        if not src.is_file():
            errors.append((rel, "ya no está en la carpeta"))
            continue
        dest = dest_dir / src.name
        n = 2
        while dest.exists():
            dest = dest_dir / f"{src.stem}_{n}{src.suffix}"
            n += 1
        try:
            if rel in copy_only:
                shutil.copy2(str(src), str(dest))
            else:
                shutil.move(str(src), str(dest))
            moved.append(rel)
        except OSError as e:
            errors.append((rel, str(e)))
    return moved, errors
