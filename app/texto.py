"""Separador de memes / capturas de texto: revisa una carpeta y marca las
imágenes con MUCHO texto (memes de Insta, capturas de páginas de filosofía o
psicología, screenshots de chat), para apartarlas a una subcarpeta _texto y que
no ensucien la galería de personas.

No LEE el texto (eso sería OCR, otra liga y otra instalación): solo detecta que
HAY texto y cuánto, con OpenCV puro —que ya está instalado— así que no baja nada
ni depende de internet. La firma del texto: bordes de caracteres que, juntados en
horizontal, forman líneas anchas y llenas. Una foto normal casi no tiene de eso;
un meme o una captura, un montón.

Igual que el revisor NSFW: nada se mueve solo. La web muestra las sospechosas con
su % de cobertura y el usuario confirma cuáles van a _texto.
"""

import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from app import DATA_DIR, IMAGE_EXTENSIONS, NSFW_DIR_NAME, TEXTO_DIR_NAME

TEXTO_THUMBS = DATA_DIR / "texto_thumbs"   # miniaturas de la última revisión
THUMB_SIZE = 260

# Umbral de COBERTURA (fracción de la imagen que es texto) para marcar una imagen.
# Una captura de página de texto ronda 0.30+; un meme 0.10-0.25; una foto normal
# con un cartelito, <0.04. El deslizador de la web lo mueve entre LO y HI.
UMBRAL = 0.10
UMBRAL_LO = 0.04   # extremo "sensible": atrapa más (incluso poco texto)
UMBRAL_HI = 0.22   # extremo "estricto": solo lo muy cargado de texto


def cobertura_texto(image_path):
    """Fracción [0..1] de la imagen cubierta por regiones tipo texto. Alto = mucho
    texto. Devuelve 0.0 si la imagen no se puede leer."""
    try:
        # np.fromfile + imdecode en vez de cv2.imread: soporta rutas con acentos/
        # unicode en Windows, que imread no abre.
        buf = np.fromfile(str(image_path), dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception:
        img = None
    if img is None:
        return 0.0
    h, w = img.shape[:2]
    if h < 8 or w < 8:
        return 0.0
    # normalizar tamaño: velocidad + que el umbral signifique lo mismo en toda foto
    esc = 1000.0 / max(h, w)
    if esc < 1:
        img = cv2.resize(img, (max(1, int(w * esc)), max(1, int(h * esc))),
                         interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gradiente morfológico: los caracteres son casi puro borde, esto los resalta
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    grad = cv2.morphologyEx(gris, cv2.MORPH_GRADIENT, k)
    _, bw = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # cerrar en horizontal: junta letras vecinas en una LÍNEA de texto
    kh = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
    lineas = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kh)
    cont, _ = cv2.findContours(lineas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area_txt = 0
    area_img = float(h * w)
    for c in cont:
        _, _, cw, ch = cv2.boundingRect(c)
        if ch < 8 or cw < 12:            # muy chico: ruido
            continue
        if ch > h * 0.33:                # muy alto para ser una línea de texto
            continue
        if cw / float(ch) < 2.0:         # las líneas de texto son anchas, no cuadradas
            continue
        if cv2.contourArea(c) / float(cw * ch) < 0.35:   # una línea llena bastante su caja
            continue
        area_txt += cw * ch
    return min(1.0, area_txt / area_img)


def scan_texto(folder, umbral=UMBRAL, progress=None, should_stop=None):
    """Revisa las imágenes de la carpeta (y subcarpetas, salvo _texto y _nsfw).

    umbral: cobertura mínima de texto para marcar (más bajo = atrapa más).

    Devuelve (results, escaneadas): results = [{"rel","cobertura","thumb"}]
    ordenado por cobertura descendente. Miniaturas en data/texto_thumbs.
    """
    folder = Path(folder)
    saltar = {NSFW_DIR_NAME, TEXTO_DIR_NAME}
    fotos = [p for p in sorted(folder.rglob("*"))
             if p.suffix.lower() in IMAGE_EXTENSIONS
             and not (set(p.relative_to(folder).parts) & saltar)]
    if not fotos:
        raise ValueError(f"No hay imágenes en {folder}")

    if TEXTO_THUMBS.is_dir():
        shutil.rmtree(TEXTO_THUMBS, ignore_errors=True)
    TEXTO_THUMBS.mkdir(parents=True, exist_ok=True)

    results, scanned = [], 0
    for i, path in enumerate(fotos, 1):
        if should_stop and should_stop():
            break
        rel = path.relative_to(folder).as_posix()
        if progress:
            progress(i, len(fotos), rel)
        scanned += 1
        try:
            cob = cobertura_texto(path)
        except Exception:
            continue   # ilegible o formato raro: se saltea
        if cob < umbral:
            continue
        thumb_name = f"{len(results):05d}.jpg"
        try:
            img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
            img.thumbnail((THUMB_SIZE, THUMB_SIZE))
            img.save(TEXTO_THUMBS / thumb_name, quality=85)
        except Exception:
            thumb_name = ""
        results.append({"rel": rel, "cobertura": round(cob, 3), "thumb": thumb_name})

    results.sort(key=lambda r: r["cobertura"], reverse=True)
    return results, scanned


def move_to_texto(folder, rels):
    """Mueve los archivos confirmados a <carpeta>/_texto/ (aplanando subcarpetas y
    evitando choques de nombre). Devuelve (movidos, errores)."""
    folder = Path(folder)
    dest_dir = folder / TEXTO_DIR_NAME
    dest_dir.mkdir(parents=True, exist_ok=True)

    moved, errors = [], []
    for rel in rels:
        src = folder / rel
        if not src.is_file():
            errors.append((rel, "ya no está en la carpeta"))
            continue
        base = rel.replace("/", "_")
        dest = dest_dir / base
        n = 2
        while dest.exists():
            dest = dest_dir / f"{Path(base).stem}_{n}{Path(base).suffix}"
            n += 1
        try:
            shutil.move(str(src), str(dest))
            moved.append(rel)
        except OSError as e:
            errors.append((rel, str(e)))
    return moved, errors
