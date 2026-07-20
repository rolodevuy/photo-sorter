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
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from app import (GROUP_DIR_NAME, IMAGE_EXTENSIONS, KNOWN_PATH, KNOWN_THUMBS,
                 NSFW_DIR_NAME, REVIEW_DIR_NAME, VIDEO_EXTENSIONS)
from app.facedet import detect_and_encode

# Distancia coseno máxima para dar por conocida a una persona. Más bajo = más
# estricto (menos autoetiquetas equivocadas, pero reconoce menos). Se usa un
# valor más estricto que el del agrupado para no poner nombres errados solos.
MATCH_EPS = 0.50

_INVALID = '<>:"/\\|?*'

# El importador por carpeta SOLO copia (no detecta caras) y las fotos se ven en
# el navegador, así que acepta un poco más que el analizador: .jfif (es JPEG) y
# .avif (los muestran los navegadores modernos). HEIC/TIFF quedan afuera porque
# el navegador no los dibuja (se copiarían pero se verían rotos).
IMPORT_EXTENSIONS = IMAGE_EXTENSIONS | {".jfif", ".avif"}


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


def known_summary(known_dir=None):
    """Lista de personas conocidas: [{name, count, faces, thumb}] ordenada por
    nombre. Si se pasa known_dir, incluye TAMBIÉN a las personas que solo existen
    como carpeta en la Biblioteca (importadas sin analizar caras): esas van con
    faces=False y count=0."""
    db = load_known()
    names = set(db)
    if known_dir:
        names |= set(library_people(known_dir))
    out = []
    for name in sorted(names, key=str.lower):
        thumb = f"{safe_name(name)}.jpg"
        out.append({
            "name": name,
            "count": int(len(db[name])) if name in db else 0,
            "faces": name in db,          # True = tiene cara aprendida (se reconoce sola)
            "thumb": thumb if (KNOWN_THUMBS / thumb).is_file() else None,
        })
    return out


def library_people(known_dir):
    """Nombres de las subcarpetas de la Biblioteca que tienen al menos una imagen.
    Sirve para que la galería muestre también a las personas importadas por
    carpeta (sin cara aprendida, por eso no están en la base de vectores)."""
    root = Path(known_dir) if known_dir else None
    if not root or not root.is_dir():
        return []
    out = []
    for sub in root.iterdir():
        if not sub.is_dir() or sub.name.startswith("_"):
            continue
        if any(p.suffix.lower() in IMAGE_EXTENSIONS for p in sub.rglob("*")):
            out.append(sub.name)
    return out


def import_folder_as_person(known_dir, src, name, recursive=True):
    """Copia TODAS las imágenes de `src` a <Biblioteca>/<name>/ y le pone una
    portada (miniatura de la primera foto, SIN detectar caras). No analiza nada:
    asume que todas las fotos son de esa persona. Devuelve cuántas copió.

    Saltea archivos que ya estén en el destino con el mismo nombre y tamaño (así
    re-importar la misma carpeta no duplica). Lanza ValueError si falta la
    Biblioteca, la carpeta origen no existe, o no hay nombre."""
    if not known_dir:
        raise ValueError("Falta configurar la Biblioteca (Ajustes).")
    src = Path(src)
    if not src.is_dir():
        raise ValueError("Esa carpeta de origen no existe.")
    name = (name or src.name).strip()
    if not name:
        raise ValueError("Falta el nombre de la persona.")
    # las carpetas de persona usan el nombre CRUDO (igual que el organizador y
    # person_photos); safe_name es solo para el archivo de la miniatura.
    dest_dir = Path(known_dir) / name
    dest_dir.mkdir(parents=True, exist_ok=True)
    ya = {(q.name, q.stat().st_size) for q in dest_dir.iterdir() if q.is_file()}
    it = src.rglob("*") if recursive else src.iterdir()
    copiadas, dup, vistos, primera = 0, 0, 0, None
    otras = {}                                   # extensiones NO soportadas -> cuántas
    for p in sorted(it):
        if not p.is_file():
            continue
        vistos += 1
        ext = p.suffix.lower()
        if ext not in IMPORT_EXTENSIONS:
            otras[ext or "(sin extensión)"] = otras.get(ext or "(sin extensión)", 0) + 1
            continue
        if (p.name, p.stat().st_size) in ya:      # ya importada antes: saltear
            dup += 1
            continue
        dest = dest_dir / p.name
        n = 2
        while dest.exists():
            dest = dest_dir / f"{p.stem}_{n}{p.suffix}"
            n += 1
        shutil.copy2(str(p), str(dest))
        ya.add((dest.name, dest.stat().st_size))
        copiadas += 1
        if primera is None:
            primera = dest
    # portada de cortesía (miniatura de la 1ra foto), solo si no tenía ya una
    thumb = KNOWN_THUMBS / f"{safe_name(name)}.jpg"
    if primera is not None and not thumb.is_file():
        try:
            KNOWN_THUMBS.mkdir(parents=True, exist_ok=True)
            im = ImageOps.exif_transpose(Image.open(primera)).convert("RGB")
            im.thumbnail((220, 220))
            im.save(thumb, quality=85)
        except Exception:
            pass
    return {"copiadas": copiadas, "dup": dup, "vistos": vistos,
            "otras": sorted(otras.items(), key=lambda x: -x[1])}


def save_uploaded_photos(known_dir, name, files):
    """Guarda archivos SUBIDOS directo (desde el navegador) en
    <Biblioteca>/<name>/. Todos se asumen de esa persona, sin analizar caras.
    Devuelve {copiadas, dup, otras}. Lanza ValueError si falta biblioteca/nombre.

    `files` son los FileStorage de request.files.getlist. Si ya hay un archivo
    con ese nombre, no se pisa: se guarda como nombre_2, _3…"""
    if not known_dir:
        raise ValueError("Falta configurar la Biblioteca (Ajustes).")
    name = (name or "").strip()
    if not name:
        raise ValueError("Falta el nombre de la persona.")
    dest_dir = Path(known_dir) / name
    dest_dir.mkdir(parents=True, exist_ok=True)
    copiadas, dup, primera = 0, 0, None
    otras = {}
    for f in files:
        fn = Path((f.filename or "")).name        # sin ruta, solo el nombre
        if not fn:
            continue
        ext = Path(fn).suffix.lower()
        if ext not in IMPORT_EXTENSIONS:
            otras[ext or "(sin extensión)"] = otras.get(ext or "(sin extensión)", 0) + 1
            continue
        dest = dest_dir / fn
        n = 2
        while dest.exists():
            dest = dest_dir / f"{Path(fn).stem}_{n}{ext}"
            n += 1
        try:
            f.save(str(dest))
        except Exception:
            continue
        copiadas += 1
        if primera is None:
            primera = dest
    # portada de cortesía si la persona todavía no tenía una
    thumb = KNOWN_THUMBS / f"{safe_name(name)}.jpg"
    if primera is not None and not thumb.is_file():
        try:
            KNOWN_THUMBS.mkdir(parents=True, exist_ok=True)
            im = ImageOps.exif_transpose(Image.open(primera)).convert("RGB")
            im.thumbnail((220, 220))
            im.save(thumb, quality=85)
        except Exception:
            pass
    return {"copiadas": copiadas, "dup": dup,
            "otras": sorted(otras.items(), key=lambda x: -x[1])}


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
    ranked = identify_ranked(db, embeddings, eps=eps, top_n=1)
    if ranked:
        return ranked[0]
    return None, 1.0


def identify_ranked(db, embeddings, eps=MATCH_EPS, top_n=3):
    """Devuelve hasta top_n candidatos [(nombre, distancia), ...] ordenados del
    más parecido al menos, dentro del umbral eps. Sirve para ofrecer el
    siguiente candidato cuando el usuario dice que la sugerencia no es correcta."""
    if not db or len(embeddings) == 0:
        return []
    group = np.asarray(embeddings, dtype=np.float32).mean(axis=0)
    n = np.linalg.norm(group)
    if n > 0:
        group = group / n

    scored = []
    for name, cent in _centroids(db).items():
        dist = 1.0 - float(np.dot(group, cent))  # distancia coseno
        if dist <= eps:
            scored.append((name, dist))
    scored.sort(key=lambda x: x[1])
    return scored[:top_n]


def _largest_face(dets):
    """De las caras de una imagen, la de mayor área (probable sujeto)."""
    def area(box):
        top, right, bottom, left = box
        return (bottom - top) * (right - left)
    return max(dets, key=lambda d: area(d[0]))


def person_photos(known_dir, name, only_nsfw=False, media="images"):
    """Fotos de <carpeta_importada>/<nombre>/ (rutas relativas a esa subcarpeta),
    para la galería de la persona. Lista vacía si la subcarpeta no existe.

    Siempre excluye _revisar. Por defecto también excluye _nsfw (las apartadas
    por el revisor); con only_nsfw=True devuelve SOLO esas, para el botón de
    mostrar/ocultar NSFW en la galería.

    media: "images" (por defecto, solo fotos — así el etiquetado/pie no toca
    videos), "videos" (solo videos) o "all" (ambos)."""
    exts = {"videos": VIDEO_EXTENSIONS,
            "all": IMAGE_EXTENSIONS | VIDEO_EXTENSIONS}.get(media, IMAGE_EXTENSIONS)
    folder = Path(known_dir) / name
    if not folder.is_dir():
        return []
    out = []
    for p in sorted(folder.rglob("*")):
        if p.suffix.lower() not in exts or REVIEW_DIR_NAME in p.parts:
            continue
        in_nsfw = NSFW_DIR_NAME in p.parts
        if in_nsfw != only_nsfw:
            continue
        out.append(p.relative_to(folder).as_posix())
    return out


def move_photo(library, from_name, rel, to_name=None):
    """Saca una foto de la carpeta de una persona.

    to_name: nombre de otra persona (se muda a su carpeta) o None (va a
    <persona>/_revisar/, apartada). Devuelve la ruta destino.
    Lanza ValueError si la foto no existe.
    """
    src = Path(library) / from_name / rel
    if not src.is_file():
        raise ValueError("Esa foto ya no está.")

    if to_name:
        dest_dir = Path(library) / to_name
    else:
        dest_dir = Path(library) / from_name / REVIEW_DIR_NAME
    dest_dir.mkdir(parents=True, exist_ok=True)

    dest = dest_dir / src.name
    n = 2
    while dest.exists():
        dest = dest_dir / f"{src.stem}_{n}{src.suffix}"
        n += 1
    shutil.move(str(src), str(dest))
    return dest


def set_cover(name, photo_path, face_index=None):
    """Elige la portada de una persona a partir de una foto suya.

    Con una sola cara (o con face_index elegido en el selector) recorta esa
    cara; sin caras usa la foto entera achicada. Devuelve (estado, encoding):
    - ("multi", cantidad): hay varias caras y no se indicó cuál — el caller
      muestra el selector para que el usuario señale a su persona.
    - ("ok", encoding): recortó la cara; el encoding sirve para sumarla a la
      base como ejemplo confirmado.
    - ("whole", None): sin caras detectadas, foto entera.
    """
    img = np.asarray(ImageOps.exif_transpose(Image.open(photo_path)).convert("RGB"))
    dets = detect_and_encode(img)
    if not dets:
        KNOWN_THUMBS.mkdir(parents=True, exist_ok=True)
        im = Image.fromarray(img)
        im.thumbnail((220, 220))
        im.save(KNOWN_THUMBS / f"{safe_name(name)}.jpg", quality=85)
        return "whole", None
    if face_index is None:
        if len(dets) > 1:
            return "multi", len(dets)
        face_index = 0
    if not 0 <= face_index < len(dets):
        raise ValueError("Esa cara ya no está en la foto (volvé a elegirla).")
    box, enc = dets[face_index]
    save_person_thumb(name, img, box)
    return "ok", enc


def save_face_crops(photo_path, out_dir, size=160, margin=0.25):
    """Recorta todas las caras de una foto a <out_dir>/0.jpg, 1.jpg... (en el
    mismo orden que devuelve el detector, para que el selector de portada y
    set_cover coincidan). Devuelve la cantidad de caras."""
    img = np.asarray(ImageOps.exif_transpose(Image.open(photo_path)).convert("RGB"))
    dets = detect_and_encode(img)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.jpg"):
        old.unlink()
    h, w = img.shape[:2]
    for i, (box, _) in enumerate(dets):
        top, right, bottom, left = box
        my = int((bottom - top) * margin)
        mx = int((right - left) * margin)
        crop = img[max(0, top - my):min(h, bottom + my),
                   max(0, left - mx):min(w, right + mx)]
        im = Image.fromarray(crop)
        im.thumbnail((size, size))
        im.save(out_dir / f"{i}.jpg", quality=85)
    return len(dets)


def enroll_from_folder(folder, progress=None, log=print):
    """Aprende personas desde una carpeta ya ordenada: cada subcarpeta es el
    nombre de una persona y sus imágenes son ejemplos de esa persona.

    Toma la cara más grande de cada imagen. Saltea _revisar (fotos donde la
    persona no está) y _grupo (fotos con varias caras, donde "la más grande"
    puede no ser la persona): así solo aprende de fotos inequívocas.
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
    # _revisar (la persona no está) y _grupo (varias caras) quedan afuera:
    # solo se aprende de fotos donde la cara más grande es, sin dudas, la persona
    skip = {REVIEW_DIR_NAME, GROUP_DIR_NAME}
    all_imgs = [(d.name, p) for d in person_dirs
                for p in sorted(d.rglob("*"))
                if p.suffix.lower() in IMAGE_EXTENSIONS
                and not skip & set(p.parts)]
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
