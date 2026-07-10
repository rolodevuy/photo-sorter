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

import numpy as np
from PIL import Image, ImageOps
from sklearn.cluster import DBSCAN

from app import CLUSTERS_PATH, ENCODINGS_NPY, FACES_DIR, IMAGE_EXTENSIONS, LABELS_PATH
from app.config import load_config
from app.facedet import detect_and_encode
from app.known import identify, load_known

# Distancia coseno máxima entre vectores SFace para considerarlos la misma
# persona. SFace considera "misma persona" con similitud coseno > 0.363, o sea
# distancia < 0.637. Usamos algo un poco más estricto para no mezclar personas
# distintas; si una persona queda partida en varios grupos, les ponés el mismo
# nombre y se unen al organizar.
EPS = 0.60

THUMBNAIL_SIZE = 160  # px del lado mayor de la miniatura
BOX_MARGIN = 0.25     # margen extra alrededor del rostro al recortar


def _load_rgb(path):
    """Carga una imagen como numpy RGB, respetando la orientación EXIF."""
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    return np.asarray(img)


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
    labels = DBSCAN(eps=EPS, min_samples=2, metric="cosine").fit_predict(
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


def run_analysis(photos_dir, exclude=None, progress=None, log=print):
    """Analiza la carpeta de origen completa y escribe clusters.json.

    Usa YuNet (detección) + SFace (reconocimiento) de OpenCV, que detectan bien
    caras anguladas o de perfil y son rápidos en CPU.
    progress: callback opcional progress(actual, total, nombre_foto).
    Devuelve (cantidad_rostros, cantidad_grupos). Lanza ValueError si no hay
    fotos o no se encuentra ningún rostro.
    """
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
        try:
            image = _load_rgb(photo)
        except Exception as e:
            log(f"no se pudo leer: {e}")
            no_faces.append(rel)
            continue
        dets = detect_and_encode(image)

        if not dets:
            log("sin rostros")
            no_faces.append(rel)
            continue
        log(f"{len(dets)} rostro(s)")

        for box, encoding in dets:
            face_id = f"{counter:05d}"
            counter += 1
            faces[face_id] = {"photo": rel, "box": list(box)}
            encodings.append(encoding)
            save_thumbnail(image, box, face_id)

    if not faces:
        raise ValueError("No se encontró ningún rostro en las fotos.")

    clusters = cluster_faces(faces, encodings)

    # Guardar los vectores (para reconocer y para sumar a la base al nombrar).
    enc_array = np.array(encodings, dtype=np.float32)
    np.save(ENCODINGS_NPY, enc_array)

    # Reconocimiento automático: los grupos que coincidan con una persona ya
    # conocida quedan preetiquetados; el asistente solo pregunta por los demás.
    known = load_known()
    auto_labels = {}
    if known:
        for cluster in clusters:
            idxs = [int(fid) for fid in cluster["faces"]]
            name, _ = identify(known, enc_array[idxs])
            if name:
                cluster["auto"] = name
                auto_labels[str(cluster["id"])] = name

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

    # Los nombres viejos corresponden a grupos que ya no existen; se reemplazan
    # por los reconocidos automáticamente (si los hay).
    if auto_labels:
        with open(LABELS_PATH, "w", encoding="utf-8") as f:
            json.dump(auto_labels, f, ensure_ascii=False, indent=2)
    else:
        LABELS_PATH.unlink(missing_ok=True)

    if auto_labels:
        log(f"[analyze] {len(auto_labels)} grupo(s) reconocido(s) automáticamente.")
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
