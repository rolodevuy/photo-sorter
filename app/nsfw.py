"""Detector NSFW local: revisa una carpeta de fotos y marca las que tienen
desnudez explícita, para revisarlas y apartarlas a una subcarpeta _nsfw.

Usa NudeNet (modelo ONNX local, sin internet). Solo cuentan las clases
inequívocas (desnudez explícita): una foto de playa con bikini NO debería
saltar. Igual nada se mueve solo: la web muestra las sospechosas con su
puntaje y el usuario confirma cuáles van a _nsfw.
"""

import shutil
from pathlib import Path

from PIL import Image, ImageOps

from app import DATA_DIR, IMAGE_EXTENSIONS, NSFW_DIR_NAME

NSFW_THUMBS = DATA_DIR / "nsfw_thumbs"   # miniaturas de la última revisión

# Clases de NudeNet que cuentan como desnudez EXPLÍCITA (partes expuestas).
# Caras, pies, panza, axilas, pecho masculino: se ignoran siempre.
EXPLICIT_CLASSES = {
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "ANUS_EXPOSED",
    "FEMALE_BREAST_EXPOSED",
    "BUTTOCKS_EXPOSED",
}

# Clases "cubiertas" (ropa interior, bikini, insinuante). Solo se usan con la
# sensibilidad alta: atrapan más, a costa de más falsos positivos tipo playa.
COVERED_CLASSES = {
    "FEMALE_GENITALIA_COVERED",
    "FEMALE_BREAST_COVERED",
    "BUTTOCKS_COVERED",
    "ANUS_COVERED",
}

# Confianza mínima por defecto (el deslizador de la web la ajusta).
SCORE_MIN = 0.45
SCORE_MIN_LO = 0.20   # extremo "sensible" del deslizador
SCORE_MIN_HI = 0.60   # extremo "estricto"

# Nombres legibles para la web (estilo directo elegido por el usuario)
LABELS_ES = {
    "FEMALE_GENITALIA_EXPOSED": "vagina",
    "MALE_GENITALIA_EXPOSED": "pija",
    "ANUS_EXPOSED": "ano",
    "FEMALE_BREAST_EXPOSED": "tetas",
    "BUTTOCKS_EXPOSED": "nalgas",
    "FEMALE_GENITALIA_COVERED": "vagina tapada",
    "FEMALE_BREAST_COVERED": "tetas tapadas",
    "BUTTOCKS_COVERED": "nalgas tapadas",
    "ANUS_COVERED": "ano tapado",
}

THUMB_SIZE = 260

_detector = None


def _get_detector():
    global _detector
    if _detector is None:
        from nudenet import NudeDetector   # import perezoso: tarda en cargar
        _detector = NudeDetector()
    return _detector


def scan_nsfw(folder, score_min=SCORE_MIN, include_covered=False,
              progress=None, should_stop=None):
    """Revisa las imágenes de la carpeta (y subcarpetas, salvo _nsfw).

    score_min: confianza mínima para marcar (más bajo = atrapa más).
    include_covered: además de lo explícito, marca lo "cubierto"/insinuante
    (ropa interior, bikini) — más recall, más falsos positivos tipo playa.

    Devuelve (results, escaneadas): results = [{"rel", "score", "labels",
    "thumb"}] ordenado por puntaje descendente. Miniaturas en data/nsfw_thumbs.
    """
    wanted = EXPLICIT_CLASSES | COVERED_CLASSES if include_covered else EXPLICIT_CLASSES

    folder = Path(folder)
    # Saltea la SUBCARPETA _nsfw (fotos ya apartadas), pero mirando la ruta
    # relativa: si apuntás DIRECTO a un _nsfw (para probar), no se excluye nada.
    photos = [p for p in sorted(folder.rglob("*"))
              if p.suffix.lower() in IMAGE_EXTENSIONS
              and NSFW_DIR_NAME not in p.relative_to(folder).parts]
    if not photos:
        raise ValueError(f"No hay imágenes en {folder}")

    detector = _get_detector()

    if NSFW_THUMBS.is_dir():
        shutil.rmtree(NSFW_THUMBS, ignore_errors=True)
    NSFW_THUMBS.mkdir(parents=True, exist_ok=True)

    results = []
    scanned = 0
    for i, path in enumerate(photos, 1):
        if should_stop and should_stop():
            break
        rel = path.relative_to(folder).as_posix()
        if progress:
            progress(i, len(photos), rel)
        scanned += 1
        try:
            dets = detector.detect(str(path))
        except Exception:
            continue   # ilegible o formato no soportado: se saltea

        hits = [d for d in dets
                if d["class"] in wanted and d["score"] >= score_min]
        if not hits:
            continue

        thumb_name = f"{len(results):05d}.jpg"
        try:
            img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
            img.thumbnail((THUMB_SIZE, THUMB_SIZE))
            img.save(NSFW_THUMBS / thumb_name, quality=85)
        except Exception:
            thumb_name = ""
        results.append({
            "rel": rel,
            "score": round(max(d["score"] for d in hits), 2),
            "labels": sorted({LABELS_ES.get(d["class"], d["class"]) for d in hits}),
            "thumb": thumb_name,
        })

    results.sort(key=lambda r: r["score"], reverse=True)
    return results, scanned


def move_to_nsfw(folder, rels):
    """Mueve los archivos confirmados a <carpeta>/_nsfw/ (aplanando subcarpetas
    y evitando choques de nombre). Devuelve (movidos, errores)."""
    folder = Path(folder)
    dest_dir = folder / NSFW_DIR_NAME
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
