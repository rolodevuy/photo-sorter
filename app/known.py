"""Base de personas conocidas: recuerda caras ya nombradas para reconocerlas
automáticamente en futuros análisis.

La base es data/known_people.json con la forma {nombre: [vector, vector, ...]},
donde cada vector es un embedding SFace (normalizado). Se puede:

- Importar desde una carpeta ya ordenada (subcarpetas = nombres de persona),
  así no hay que reetiquetar gente que ya clasificaste.
- Sumar caras de un grupo recién nombrado en la web.
- Identificar a qué persona conocida corresponde un grupo nuevo.

Todo local, sin conexión.
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from app import IMAGE_EXTENSIONS, KNOWN_PATH, KNOWN_THUMBS
from app.facedet import detect_and_encode

# Distancia coseno máxima para dar por conocida a una persona. Más bajo = más
# estricto (menos autoetiquetas equivocadas, pero reconoce menos). Se usa un
# valor más estricto que el del agrupado para no poner nombres errados solos.
MATCH_EPS = 0.50

_INVALID = '<>:"/\\|?*'


def safe_name(name):
    """Nombre válido como archivo (para la miniatura de la persona)."""
    return "".join(c for c in name if c not in _INVALID).strip() or "_"


def save_person_thumb(name, image_rgb, box, size=220, margin=0.2):
    """Guarda la miniatura representativa de una persona (recorte del rostro)."""
    top, right, bottom, left = box
    h, w = image_rgb.shape[:2]
    my = int((bottom - top) * margin)
    mx = int((right - left) * margin)
    crop = image_rgb[max(0, top - my):min(h, bottom + my),
                     max(0, left - mx):min(w, right + mx)]
    if crop.size == 0:
        return
    KNOWN_THUMBS.mkdir(parents=True, exist_ok=True)
    im = Image.fromarray(crop)
    im.thumbnail((size, size))
    im.save(KNOWN_THUMBS / f"{safe_name(name)}.jpg", quality=85)


def known_summary():
    """Lista de personas conocidas: [{name, count, thumb}] ordenada por nombre."""
    db = load_known()
    out = []
    for name in sorted(db, key=str.lower):
        thumb = f"{safe_name(name)}.jpg"
        out.append({
            "name": name,
            "count": int(len(db[name])),
            "thumb": thumb if (KNOWN_THUMBS / thumb).is_file() else None,
        })
    return out


def forget_person(name):
    """Borra una persona de la base y su miniatura."""
    db = load_known()
    if name in db:
        del db[name]
        save_known(db)
    thumb = KNOWN_THUMBS / f"{safe_name(name)}.jpg"
    thumb.unlink(missing_ok=True)


def load_known():
    """Devuelve {nombre: np.ndarray (N, D)} con los vectores de cada persona."""
    if not KNOWN_PATH.is_file():
        return {}
    with open(KNOWN_PATH, encoding="utf-8") as f:
        raw = json.load(f)
    return {name: np.array(vecs, dtype=np.float32) for name, vecs in raw.items() if vecs}


def save_known(db):
    KNOWN_PATH.parent.mkdir(parents=True, exist_ok=True)
    serializable = {name: np.asarray(vecs, dtype=np.float32).tolist() for name, vecs in db.items()}
    with open(KNOWN_PATH, "w", encoding="utf-8") as f:
        json.dump(serializable, f)


def add_faces(db, name, embeddings, max_per_person=60):
    """Agrega vectores a una persona (limita cuántos guarda por persona)."""
    if not embeddings:
        return db
    new = np.asarray(embeddings, dtype=np.float32)
    if name in db and len(db[name]):
        db[name] = np.vstack([db[name], new])
    else:
        db[name] = new
    # si se acumulan demasiados, quedarse con una muestra
    if len(db[name]) > max_per_person:
        idx = np.linspace(0, len(db[name]) - 1, max_per_person).astype(int)
        db[name] = db[name][idx]
    return db


def _centroids(db):
    """Vector promedio (normalizado) de cada persona."""
    cents = {}
    for name, vecs in db.items():
        c = vecs.mean(axis=0)
        n = np.linalg.norm(c)
        if n > 0:
            c = c / n
        cents[name] = c
    return cents


def identify(db, embeddings, eps=MATCH_EPS):
    """Dado el conjunto de vectores de un grupo, devuelve (nombre, distancia)
    de la persona conocida más parecida, o (None, dist) si ninguna alcanza."""
    if not db or len(embeddings) == 0:
        return None, 1.0
    group = np.asarray(embeddings, dtype=np.float32).mean(axis=0)
    n = np.linalg.norm(group)
    if n > 0:
        group = group / n

    best_name, best_dist = None, 2.0
    for name, cent in _centroids(db).items():
        dist = 1.0 - float(np.dot(group, cent))  # distancia coseno
        if dist < best_dist:
            best_name, best_dist = name, dist
    if best_dist <= eps:
        return best_name, best_dist
    return None, best_dist


def _largest_face(dets):
    """De las caras de una imagen, la de mayor área (probable sujeto)."""
    def area(box):
        top, right, bottom, left = box
        return (bottom - top) * (right - left)
    return max(dets, key=lambda d: area(d[0]))


def enroll_from_folder(folder, progress=None, log=print):
    """Aprende personas desde una carpeta ya ordenada: cada subcarpeta es el
    nombre de una persona y sus imágenes son ejemplos de esa persona.

    Toma la cara más grande de cada imagen (para evitar acompañantes).
    Devuelve dict {nombre: cantidad_de_caras_agregadas}.
    """
    folder = Path(folder)
    person_dirs = [d for d in sorted(folder.iterdir()) if d.is_dir()]
    if not person_dirs:
        raise ValueError("Esa carpeta no tiene subcarpetas de personas.")

    db = load_known()
    added = {}
    best_area = {}   # nombre -> área de la cara más grande vista (para la miniatura)
    # total de imágenes para el progreso
    all_imgs = [(d.name, p) for d in person_dirs
                for p in sorted(d.rglob("*")) if p.suffix.lower() in IMAGE_EXTENSIONS]
    for i, (name, path) in enumerate(all_imgs, 1):
        if progress:
            progress(i, len(all_imgs), f"{name}/{path.name}")
        try:
            img = np.asarray(ImageOps.exif_transpose(Image.open(path)).convert("RGB"))
            dets = detect_and_encode(img)
        except Exception as e:
            log(f"[known] no se pudo leer {path.name}: {e}")
            continue
        if not dets:
            continue
        box, emb = _largest_face(dets)
        add_faces(db, name, [emb])
        added[name] = added.get(name, 0) + 1
        # guardar como miniatura la cara más grande (más clara) de la persona
        top, right, bottom, left = box
        area = (bottom - top) * (right - left)
        if area > best_area.get(name, 0):
            best_area[name] = area
            save_person_thumb(name, img, box)

    save_known(db)
    return added
