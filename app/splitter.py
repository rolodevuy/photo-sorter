"""Separador por cantidad de caras: encuentra las fotos donde hay MÁS DE UNA
persona y las aparta a una subcarpeta _grupo.

Para qué sirve: el importador aprende de cada foto "la cara más grande". Eso
es infalible cuando la persona está sola, pero en una foto grupal puede
aprender la cara equivocada (contaminando la base). Separando las grupales:

- Las fotos de una sola cara quedan donde están -> el import aprende de ellas
  sin riesgo.
- Las grupales van a _grupo, que el import saltea -> se procesan aparte con el
  corrector, que pregunta cuál cara es la persona.

Clave: solo se cuentan las caras GRANDES (tamaño mínimo configurable). Así una
selfie en la calle con transeúntes de fondo NO se toma como foto grupal.
"""

import shutil
from pathlib import Path

from PIL import Image, ImageOps

from app import DATA_DIR, GROUP_DIR_NAME, IMAGE_EXTENSIONS, NSFW_DIR_NAME, REVIEW_DIR_NAME
from app.analyze import MIN_FACE_PX, _big_enough, _load_rgb
from app.facedet import detect_and_encode

SPLIT_THUMBS = DATA_DIR / "split_thumbs"   # miniaturas de la última separación
THUMB_SIZE = 220


def scan_groups(folder, min_px=MIN_FACE_PX, progress=None, should_stop=None):
    """Cuenta las caras grandes de cada foto de la carpeta.

    Devuelve (grupales, resumen): grupales = [{"rel", "faces", "thumb"}] de las
    fotos con 2+ caras (las candidatas a mover), ordenadas por cantidad de
    caras. resumen = {"escaneadas", "solas", "grupales", "sin_cara"}.
    """
    folder = Path(folder)
    photos = [p for p in sorted(folder.rglob("*"))
              if p.suffix.lower() in IMAGE_EXTENSIONS
              and not {GROUP_DIR_NAME, REVIEW_DIR_NAME, NSFW_DIR_NAME} & set(p.parts)]
    if not photos:
        raise ValueError(f"No hay imágenes en {folder}")

    if SPLIT_THUMBS.is_dir():
        shutil.rmtree(SPLIT_THUMBS, ignore_errors=True)
    SPLIT_THUMBS.mkdir(parents=True, exist_ok=True)

    grupales = []
    solas = sin_cara = escaneadas = 0
    for i, path in enumerate(photos, 1):
        if should_stop and should_stop():
            break
        rel = path.relative_to(folder).as_posix()
        if progress:
            progress(i, len(photos), rel)
        escaneadas += 1
        try:
            img = _load_rgb(path)
            # solo caras grandes: los transeúntes de fondo no cuentan
            n = sum(1 for box, _ in detect_and_encode(img) if _big_enough(box, min_px))
        except Exception:
            continue   # ilegible: se saltea sin molestar

        if n == 0:
            sin_cara += 1
        elif n == 1:
            solas += 1
        else:
            thumb_name = f"{len(grupales):05d}.jpg"
            try:
                im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
                im.thumbnail((THUMB_SIZE, THUMB_SIZE))
                im.save(SPLIT_THUMBS / thumb_name, quality=85)
            except Exception:
                thumb_name = ""
            grupales.append({"rel": rel, "faces": n, "thumb": thumb_name})

    grupales.sort(key=lambda g: g["faces"], reverse=True)
    resumen = {"escaneadas": escaneadas, "solas": solas,
               "grupales": len(grupales), "sin_cara": sin_cara}
    return grupales, resumen


def move_to_group(folder, rels):
    """Mueve cada foto a un _grupo AL LADO de donde está (relativo a su carpeta),
    no a un _grupo único de la raíz. Así, al escanear la carpeta general, la
    grupal de cada persona va a <persona>/_grupo/ (queda con su persona, y el
    import de esa persona la saltea). Una foto suelta en la raíz va a la raíz/_grupo.
    Devuelve (movidas, errores)."""
    folder = Path(folder)
    moved, errors = [], []
    for rel in rels:
        src = folder / rel
        if not src.is_file():
            errors.append((rel, "ya no está en la carpeta"))
            continue
        dest_dir = src.parent / GROUP_DIR_NAME   # _grupo junto a la foto
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / src.name
        n = 2
        while dest.exists():
            dest = dest_dir / f"{src.stem}_{n}{src.suffix}"
            n += 1
        try:
            shutil.move(str(src), str(dest))
            moved.append(rel)
        except OSError as e:
            errors.append((rel, str(e)))
    return moved, errors
