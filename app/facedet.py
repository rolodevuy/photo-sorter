"""Motor de detección y reconocimiento facial con OpenCV (YuNet + SFace).

Reemplaza a dlib/face_recognition: YuNet detecta las caras (mucho mejor con
rostros de perfil o anguladas) y SFace calcula un vector por cara para agrupar
personas. Todo local; los modelos ONNX están en app/models/.

Función principal: detect_and_encode(image_rgb) -> lista de (box, embedding),
donde box = (top, right, bottom, left) en el mismo tamaño de la imagen dada, y
embedding es un vector normalizado (comparable por distancia coseno).
"""

from pathlib import Path

import cv2
import numpy as np

MODEL_DIR = Path(__file__).resolve().parent / "models"
YUNET_PATH = MODEL_DIR / "yunet.onnx"
SFACE_PATH = MODEL_DIR / "sface.onnx"

# Confianza mínima de YuNet para aceptar una cara (0-1). Más bajo = más caras
# (incluye dudosas); más alto = más estricto.
SCORE_THRESHOLD = 0.6

_detector = None
_recognizer = None


def _load():
    global _detector, _recognizer
    if _detector is None:
        if not YUNET_PATH.is_file() or not SFACE_PATH.is_file():
            raise FileNotFoundError(
                "Faltan los modelos en app/models/ (yunet.onnx y sface.onnx).")
        _detector = cv2.FaceDetectorYN.create(
            str(YUNET_PATH), "", (320, 320), score_threshold=SCORE_THRESHOLD)
        _recognizer = cv2.FaceRecognizerSF.create(str(SFACE_PATH), "")
    return _detector, _recognizer


def detect_and_encode(image_rgb):
    """Detecta y codifica todas las caras de una imagen RGB (numpy).

    Devuelve [(box, embedding), ...]. Lista vacía si no hay caras.
    """
    detector, recognizer = _load()
    img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    h, w = img_bgr.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(img_bgr)

    results = []
    if faces is None:
        return results
    for f in faces:
        aligned = recognizer.alignCrop(img_bgr, f)
        feat = recognizer.feature(aligned).flatten().astype(np.float32)
        norm = np.linalg.norm(feat)
        if norm > 0:
            feat = feat / norm  # normalizar -> comparable por distancia coseno
        x, y, bw, bh = f[0], f[1], f[2], f[3]
        top = max(0, int(y))
        left = max(0, int(x))
        bottom = min(h, int(y + bh))
        right = min(w, int(x + bw))
        results.append(((top, right, bottom, left), feat))
    return results
