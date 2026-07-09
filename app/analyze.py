"""Analyze: detecta rostros en la carpeta de origen y agrupa los parecidos.

No necesita entrenamiento previo: recorre la carpeta elegida (las fotos NO se
mueven ni se copian), calcula un encoding por cada rostro encontrado y agrupa
los que pertenecen a la misma persona (clustering con DBSCAN). Guarda:

- data/faces/<face_id>.jpg   miniatura de cada rostro (para la web)
- data/clusters.json         rostros + grupos + carpeta de origen usada

Todo corre local, sin ninguna conexión a internet.

Uso normal: desde la web (python -m app.flask_app).
También por consola (usa las carpetas guardadas): python -m app.analyze
"""

import json
import shutil
import sys
from pathlib import Path

import face_recognition
import numpy as np
from PIL import Image
from sklearn.cluster import DBSCAN

from app import CLUSTERS_PATH, FACES_DIR, IMAGE_EXTENSIONS, LABELS_PATH
from app.config import load_config

# Distancia máxima entre encodings para considerarlos la misma persona.
# Más bajo = grupos más estrictos (una persona puede partirse en varios grupos).
# Más alto = más permisivo (riesgo de mezclar personas distintas en un grupo).
EPS = 0.45

THUMBNAIL_SIZE = 160  # px del lado mayor de la miniatura
BOX_MARGIN = 0.25     # margen extra alrededor del rostro al recortar

# En modo preciso (CNN) se achica la imagen a este lado máximo antes de
# detectar: el CNN igual encuentra las caras y así es mucho más rápido y no
# se queda sin memoria con fotos grandes (1080x1920, etc.).
PRECISE_MAX_DIM = 800


def _downscale(image, max_dim):
    """Achica la imagen si su lado mayor supera max_dim. Devuelve numpy RGB."""
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return image
    scale = max_dim / longest
    new_size = (int(w * scale), int(h * scale))
    return np.asarray(Image.fromarray(image).resize(new_size, Image.LANCZOS))


def list_photos(photos_dir, exclude=None):
    """Lista las imágenes del origen, salteando la carpeta de destino."""
    photos = []
    for p in sorted(photos_dir.rglob("*")):
        if exclude and p.is_relative_to(exclude):
            continue
        if p.suffix.lower() in IMAGE_EXTENSIONS:
            photos.append(p)
    return photos


def save_thumbnail(image, box, face_id):
    """Recorta el rostro (con margen) y guarda una miniatura."""
    top, right, bottom, left = box
    h, w = image.shape[:2]
    margin_y = int((bottom - top) * BOX_MARGIN)
    margin_x = int((right - left) * BOX_MARGIN)
    crop = image[
        max(0, top - margin_y):min(h, bottom + margin_y),
        max(0, left - margin_x):min(w, right + margin_x),
    ]
    thumb = Image.fromarray(crop)
    thumb.thumbnail((THUMBNAIL_SIZE, THUMBNAIL_SIZE))
    thumb.save(FACES_DIR / f"{face_id}.jpg", quality=85)


def cluster_faces(faces, encodings):
    """Agrupa los encodings por persona. Devuelve lista de clusters."""
    face_ids = list(faces.keys())
    labels = DBSCAN(eps=EPS, min_samples=2, metric="euclidean").fit_predict(
        np.array(encodings)
    )

    groups = {}
    singles = []
    for face_id, label in zip(face_ids, labels):
        if label == -1:
            # Rostro que no matcheó con ningún otro: grupo propio de 1.
            singles.append(face_id)
        else:
            groups.setdefault(int(label), []).append(face_id)

    # Clusters grandes primero, singles al final
    return [
        {"id": i, "faces": members}
        for i, members in enumerate(
            sorted(groups.values(), key=len, reverse=True) + [[s] for s in singles]
        )
    ]


def run_analysis(photos_dir, exclude=None, progress=None, log=print, precise=False):
    """Analiza la carpeta de origen completa y escribe clusters.json.

    progress: callback opcional progress(actual, total, nombre_foto).
    precise: si True usa el modelo CNN (detecta caras anguladas/de perfil, pero
    es MUCHO más lento). Si False usa HOG (rápido, solo caras frontales).
    Devuelve (cantidad_rostros, cantidad_grupos). Lanza ValueError si no hay
    fotos o no se encuentra ningún rostro.
    """
    model = "cnn" if precise else "hog"
    # Con HOG, upsamplear 1 vez ayuda a agarrar caras algo más chicas o giradas.
    upsample = 1
    photos_dir = Path(photos_dir)
    photos = list_photos(photos_dir, exclude)
    if not photos:
        raise ValueError(f"No hay imágenes en {photos_dir}")

    # Miniaturas de corridas anteriores fuera; el análisis rehace todo.
    if FACES_DIR.is_dir():
        shutil.rmtree(FACES_DIR)
    FACES_DIR.mkdir(parents=True, exist_ok=True)

    faces = {}      # face_id -> {"photo": ruta relativa al origen, "box": [t,r,b,l]}
    encodings = []
    no_faces = []
    counter = 0

    for i, photo in enumerate(photos, 1):
        rel = photo.relative_to(photos_dir).as_posix()
        if progress:
            progress(i, len(photos), rel)
        log(f"[analyze] ({i}/{len(photos)}) {rel} ...", end=" ")
        image = face_recognition.load_image_file(photo)
        if precise:
            # achicar antes del CNN: más rápido y sin problemas de memoria
            image = _downscale(image, PRECISE_MAX_DIM)
        locations = face_recognition.face_locations(image, upsample, model)
        photo_encodings = face_recognition.face_encodings(image, locations)

        if not locations:
            log("sin rostros")
            no_faces.append(rel)
            continue
        log(f"{len(locations)} rostro(s)")

        for box, encoding in zip(locations, photo_encodings):
            face_id = f"{counter:05d}"
            counter += 1
            faces[face_id] = {"photo": rel, "box": list(box)}
            encodings.append(encoding)
            save_thumbnail(image, box, face_id)

    if not faces:
        raise ValueError("No se encontró ningún rostro en las fotos.")

    clusters = cluster_faces(faces, encodings)

    with open(CLUSTERS_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "photos_dir": str(photos_dir),
                "faces": faces,
                "clusters": clusters,
                "no_faces": no_faces,
            },
            f, ensure_ascii=False, indent=2,
        )

    # Los nombres viejos corresponden a grupos que ya no existen.
    LABELS_PATH.unlink(missing_ok=True)

    return len(faces), len(clusters)


def main():
    cfg = load_config()
    if not cfg.get("photos_dir"):
        print("[analyze] Todavía no elegiste la carpeta de origen.")
        print("[analyze] Corré la web y elegila ahí: python -m app.flask_app")
        sys.exit(1)

    photos_dir = Path(cfg["photos_dir"])
    if not photos_dir.is_dir():
        print(f"[analyze] La carpeta de origen no existe: {photos_dir}")
        sys.exit(1)

    exclude = Path(cfg["output_dir"]) if cfg.get("output_dir") else None
    try:
        n_faces, n_clusters = run_analysis(photos_dir, exclude=exclude)
    except ValueError as e:
        print(f"[analyze] {e}")
        sys.exit(1)

    print(f"[analyze] Listo: {n_faces} rostro(s) en {n_clusters} grupo(s).")
    print("[analyze] Ahora corré la web para ponerles nombre: python -m app.flask_app")


if __name__ == "__main__":
    main()
