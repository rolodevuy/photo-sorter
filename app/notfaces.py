"""Lista negra de detecciones que NO son personas (tapizados, objetos, manchas).

Cuando en la web se marca un grupo con "No es una persona", los vectores de
esas caras falsas se guardan acá (data/not_faces.json). En los análisis
siguientes, toda detección muy parecida a una de la lista se descarta sola:
el mismo tapizado no vuelve a aparecer nunca.

Se puede vaciar desde la página de personas conocidas (por si algo se
descartó por error).
"""

import json

import numpy as np

from app import NOT_FACES_PATH

# Distancia coseno máxima a un vector de la lista para descartar una
# detección. Más estricto que los umbrales de agrupado, para no descartar por
# accidente a una persona real que se parezca a algo de la lista.
NOT_FACE_EPS = 0.45

# Tope de vectores guardados; llegado el límite se van olvidando los más
# viejos (con unas pocas muestras por objeto falso ya alcanza).
MAX_VECTORS = 500


def load_not_faces():
    """Matriz (N, D) con los vectores de la lista, o None si está vacía."""
    if not NOT_FACES_PATH.is_file():
        return None
    with open(NOT_FACES_PATH, encoding="utf-8") as f:
        vecs = json.load(f)
    if not vecs:
        return None
    return np.array(vecs, dtype=np.float32)


def add_not_faces(vectors):
    """Suma vectores de detecciones falsas a la lista."""
    old = load_not_faces()
    new = np.asarray(vectors, dtype=np.float32)
    if new.ndim == 1:
        new = new[None, :]
    allv = np.vstack([old, new]) if old is not None else new
    if len(allv) > MAX_VECTORS:
        allv = allv[-MAX_VECTORS:]
    with open(NOT_FACES_PATH, "w", encoding="utf-8") as f:
        json.dump(allv.tolist(), f)


def count_not_faces():
    black = load_not_faces()
    return 0 if black is None else int(len(black))


def clear_not_faces():
    NOT_FACES_PATH.unlink(missing_ok=True)


def is_not_face(encoding, black):
    """True si la detección se parece a algo ya marcado como "no es una persona"."""
    if black is None:
        return False
    return float((1.0 - black @ encoding).min()) < NOT_FACE_EPS
