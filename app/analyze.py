"""Analyze: detecta rostros en todas las fotos y agrupa los parecidos.

No necesita entrenamiento previo: recorre data/photos/, calcula un encoding
por cada rostro encontrado y agrupa los que pertenecen a la misma persona
(clustering con DBSCAN). Guarda:

- data/faces/<face_id>.jpg   miniatura de cada rostro (para la web)
- data/clusters.json         rostros + grupos de la misma persona

Todo corre local, sin ninguna conexión a internet.

Uso:
    python -m app.analyze
"""

import json
import sys

import face_recognition
import numpy as np
from PIL import Image
from sklearn.cluster import DBSCAN

from app import CLUSTERS_PATH, FACES_DIR, IMAGE_EXTENSIONS, PHOTOS_DIR

# Distancia máxima entre encodings para considerarlos la misma persona.
# Más bajo = grupos más estrictos (una persona puede partirse en varios grupos).
# Más alto = más permisivo (riesgo de mezclar personas distintas en un grupo).
EPS = 0.45

THUMBNAIL_SIZE = 160  # px del lado mayor de la miniatura
BOX_MARGIN = 0.25     # margen extra alrededor del rostro al recortar


def iter_photos():
    for p in sorted(PHOTOS_DIR.rglob("*")):
        if p.suffix.lower() in IMAGE_EXTENSIONS:
            yield p


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


def scan_faces():
    """Detecta todos los rostros. Devuelve (faces, encodings, fotos_sin_rostro)."""
    faces = {}      # face_id -> {"photo": ruta relativa, "box": [t, r, b, l]}
    encodings = []
    no_faces = []
    counter = 0

    for photo in iter_photos():
        rel = photo.relative_to(PHOTOS_DIR).as_posix()
        print(f"[analyze] {rel} ...", end=" ")
        image = face_recognition.load_image_file(photo)
        locations = face_recognition.face_locations(image)
        photo_encodings = face_recognition.face_encodings(image, locations)

        if not locations:
            print("sin rostros")
            no_faces.append(rel)
            continue
        print(f"{len(locations)} rostro(s)")

        for box, encoding in zip(locations, photo_encodings):
            face_id = f"{counter:05d}"
            counter += 1
            faces[face_id] = {"photo": rel, "box": list(box)}
            encodings.append(encoding)
            save_thumbnail(image, box, face_id)

    return faces, encodings, no_faces


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
    clusters = [
        {"id": i, "faces": members}
        for i, members in enumerate(
            sorted(groups.values(), key=len, reverse=True) + [[s] for s in singles]
        )
    ]
    return clusters


def main():
    if not PHOTOS_DIR.is_dir():
        print(f"[analyze] No existe {PHOTOS_DIR}.")
        print("[analyze] Creá data/photos/ y poné ahí las fotos a clasificar.")
        sys.exit(1)

    FACES_DIR.mkdir(parents=True, exist_ok=True)

    faces, encodings, no_faces = scan_faces()
    if not faces:
        print("[analyze] No se encontró ningún rostro en las fotos.")
        sys.exit(1)

    clusters = cluster_faces(faces, encodings)

    with open(CLUSTERS_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {"faces": faces, "clusters": clusters, "no_faces": no_faces},
            f, ensure_ascii=False, indent=2,
        )

    multi = sum(1 for c in clusters if len(c["faces"]) > 1)
    print(f"[analyze] Listo: {len(faces)} rostro(s) en {len(clusters)} grupo(s) "
          f"({multi} con más de un rostro).")
    print(f"[analyze] Ahora corré la web para ponerles nombre: python -m app.flask_app")


if __name__ == "__main__":
    main()
