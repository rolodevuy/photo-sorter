"""Indexer: entrena con fotos de referencia.

Recorre data/training/<nombre_persona>/ y calcula un encoding facial por
cada foto de referencia. Guarda todos los encodings en data/encodings.pkl.

Uso:
    python -m app.indexer
"""

import pickle
import sys

import face_recognition

from app import ENCODINGS_PATH, IMAGE_EXTENSIONS, TRAINING_DIR


def iter_training_images():
    """Genera pares (nombre_persona, ruta_imagen) desde data/training/."""
    for person_dir in sorted(TRAINING_DIR.iterdir()):
        if not person_dir.is_dir():
            continue
        for image_path in sorted(person_dir.iterdir()):
            if image_path.suffix.lower() in IMAGE_EXTENSIONS:
                yield person_dir.name, image_path


def build_index():
    """Calcula los encodings de todas las fotos de referencia.

    Devuelve un dict {"names": [...], "encodings": [...]} donde cada
    posición i corresponde a un rostro de referencia de names[i].
    """
    names = []
    encodings = []

    for person, image_path in iter_training_images():
        print(f"[indexer] Procesando {person}: {image_path.name} ...", end=" ")
        image = face_recognition.load_image_file(image_path)
        face_encodings = face_recognition.face_encodings(image)

        if not face_encodings:
            print("SIN ROSTRO (salteada)")
            continue
        if len(face_encodings) > 1:
            print(f"AVISO: {len(face_encodings)} rostros, se usa el primero")
        else:
            print("ok")

        names.append(person)
        encodings.append(face_encodings[0])

    return {"names": names, "encodings": encodings}


def main():
    if not TRAINING_DIR.is_dir():
        print(f"[indexer] No existe {TRAINING_DIR}.")
        print("[indexer] Creá data/training/<nombre_persona>/ con fotos de referencia.")
        sys.exit(1)

    index = build_index()

    if not index["names"]:
        print("[indexer] No se encontró ningún rostro en las fotos de referencia.")
        sys.exit(1)

    ENCODINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(ENCODINGS_PATH, "wb") as f:
        pickle.dump(index, f)

    people = sorted(set(index["names"]))
    print(f"[indexer] Listo: {len(index['encodings'])} encodings de {len(people)} persona(s): {', '.join(people)}")
    print(f"[indexer] Guardado en {ENCODINGS_PATH}")


if __name__ == "__main__":
    main()
