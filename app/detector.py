"""Detector: detecta y reconoce rostros en fotos nuevas.

Recorre data/photos/, detecta rostros en cada imagen y los compara contra
los encodings entrenados por el indexer. Escribe los resultados en
data/detections.json.

Uso:
    python -m app.detector
"""

import json
import pickle
import sys

import face_recognition
import numpy as np

from app import DETECTIONS_PATH, ENCODINGS_PATH, IMAGE_EXTENSIONS, PHOTOS_DIR

# Distancia máxima para considerar que dos rostros son la misma persona.
# Más bajo = más estricto (menos falsos positivos, más "desconocidos").
TOLERANCE = 0.6


def load_index():
    with open(ENCODINGS_PATH, "rb") as f:
        return pickle.load(f)


def recognize_face(face_encoding, index):
    """Compara un rostro contra el índice. Devuelve (nombre, distancia)."""
    distances = face_recognition.face_distance(index["encodings"], face_encoding)
    best = int(np.argmin(distances))
    if distances[best] <= TOLERANCE:
        return index["names"][best], float(distances[best])
    return "desconocido", float(distances[best])


def detect_photo(image_path, index):
    """Detecta y reconoce todos los rostros de una foto."""
    image = face_recognition.load_image_file(image_path)
    locations = face_recognition.face_locations(image)
    encodings = face_recognition.face_encodings(image, locations)

    faces = []
    for (top, right, bottom, left), encoding in zip(locations, encodings):
        name, distance = recognize_face(encoding, index)
        faces.append({
            "name": name,
            "distance": round(distance, 4),
            "box": {"top": top, "right": right, "bottom": bottom, "left": left},
        })
    return faces


def main():
    if not ENCODINGS_PATH.is_file():
        print(f"[detector] No existe {ENCODINGS_PATH}. Corré primero: python -m app.indexer")
        sys.exit(1)
    if not PHOTOS_DIR.is_dir():
        print(f"[detector] No existe {PHOTOS_DIR}. Creá data/photos/ con las fotos a clasificar.")
        sys.exit(1)

    index = load_index()
    results = {}

    photos = [p for p in sorted(PHOTOS_DIR.rglob("*")) if p.suffix.lower() in IMAGE_EXTENSIONS]
    if not photos:
        print(f"[detector] No hay imágenes en {PHOTOS_DIR}.")
        sys.exit(1)

    for photo in photos:
        rel = photo.relative_to(PHOTOS_DIR).as_posix()
        print(f"[detector] {rel} ...", end=" ")
        faces = detect_photo(photo, index)
        results[rel] = faces
        if faces:
            print(", ".join(f"{f['name']} ({f['distance']})" for f in faces))
        else:
            print("sin rostros")

    with open(DETECTIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    total_faces = sum(len(v) for v in results.values())
    print(f"[detector] Listo: {total_faces} rostro(s) en {len(results)} foto(s).")
    print(f"[detector] Resultados en {DETECTIONS_PATH}")


if __name__ == "__main__":
    main()
