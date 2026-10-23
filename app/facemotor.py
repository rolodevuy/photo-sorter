"""Motor de caras con insightface (SCRFD + ArcFace), sobre onnxruntime-gpu.

Alternativa a YuNet+SFace (app/facedet.py). ArcFace R50 genera vectores de 512
dimensiones que separan mucho mejor caras parecidas: es el motor que se usa
cuando se quiere MENOS confusión entre personas.

IMPORTANTE — los embeddings NO son compatibles entre motores:
  - SFace  -> 128 dimensiones
  - ArcFace -> 512 dimensiones
Una base de conocidos (data/known_people.json) armada con un motor NO sirve
para el otro. Si se cambia de motor hay que RE-ENROLAR.

Este módulo NO decide cuál se usa: expone las funciones y quien lo llame elige.
Todo local: los pesos se bajan una sola vez a ~/.insightface y después no se
toca la red.
"""

import threading
from pathlib import Path

import numpy as np

# Nombre del paquete de modelos. "buffalo_l" = SCRFD-10G (detección) +
# ArcFace w600k_r50 (reconocimiento) + landmarks. Los pesos viven en
# %USERPROFILE%/.insightface/models/buffalo_l/ (~325 MB la primera vez).
PACK = "buffalo_l"

# Tamaño de entrada de la detección. insightface reescala la foto a este tamaño
# antes de buscar caras. 640 es el valor histórico de ArcFace; con fotos muy
# grandes y caras chicas conviene subirlo (más lento, encuentra más).
DET_SIZE = (640, 640)

# Confianza mínima de SCRFD para aceptar una cara (0-1).
DET_THRESHOLD = 0.5

_app = None
_lock = threading.Lock()
_motor_activo = None      # "arcface" | "sface" | None (se resuelve en el primer uso)


def _preparar_dlls_cuda():
    """Deja visibles las DLLs de CUDA/cuDNN del venv (ver app/cuda_dlls.py)."""
    try:
        from app import cuda_dlls
        return cuda_dlls.preparar()
    except Exception:
        return []


def disponible():
    """True si insightface se puede importar (el paquete está instalado)."""
    try:
        import insightface  # noqa: F401
        return True
    except Exception:
        return False


def _providers():
    """CUDA si se puede, CPU si no. Nunca falla: devuelve una lista válida."""
    try:
        import onnxruntime as ort
        disp = ort.get_available_providers()
        if "CUDAExecutionProvider" in disp:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    except Exception:
        pass
    return ["CPUExecutionProvider"]


def _cargar():
    """Crea la app de insightface una sola vez (thread-safe)."""
    global _app
    if _app is not None:
        return _app
    with _lock:
        if _app is not None:
            return _app
        _preparar_dlls_cuda()
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(name=PACK, providers=_providers())
        # ctx_id=0 -> GPU 0; si no hay CUDA, insightface igual arranca en CPU
        try:
            app.prepare(ctx_id=0, det_size=DET_SIZE, det_thresh=DET_THRESHOLD)
        except Exception:
            app.prepare(ctx_id=-1, det_size=DET_SIZE, det_thresh=DET_THRESHOLD)
        _app = app
        return _app


def usar_arcface():
    """Marca que el motor activo es ArcFace (para que lo consulten otros)."""
    global _motor_activo
    _motor_activo = "arcface"


def motor_activo():
    return _motor_activo


def detect_and_encode(image_rgb):
    """Detecta y codifica caras de una imagen RGB (numpy uint8).

    Devuelve [(box, embedding), ...] con el MISMO formato que
    app.facedet.detect_and_encode:
      - box = (top, right, bottom, left) en píxeles de la imagen dada
      - embedding = vector float32 normalizado (512 dims en ArcFace)

    Lanza RuntimeError si insightface no está o no puede correr, para que quien
    llame decida el fallback (no se traga el error).
    """
    app = _cargar()
    # insightface espera BGR
    img_bgr = np.ascontiguousarray(image_rgb[:, :, ::-1]) if image_rgb.ndim == 3 else image_rgb
    h, w = img_bgr.shape[:2]

    with _lock:
        faces = app.get(img_bgr)

    results = []
    for f in faces:
        emb = getattr(f, "normed_embedding", None)
        if emb is None:
            emb = f.embedding
            n = np.linalg.norm(emb)
            if n > 0:
                emb = emb / n
        emb = np.asarray(emb, dtype=np.float32).flatten()
        x1, y1, x2, y2 = f.bbox
        top = max(0, int(y1))
        left = max(0, int(x1))
        bottom = min(h, int(y2))
        right = min(w, int(x2))
        results.append(((top, right, bottom, left), emb))
    return results


if __name__ == "__main__":
    import sys
    print("insightface disponible:", disponible())
    print("providers:", _providers())
    if len(sys.argv) > 1:
        from PIL import Image, ImageOps
        img = np.asarray(ImageOps.exif_transpose(Image.open(sys.argv[1])).convert("RGB"))
        dets = detect_and_encode(img)
        print(f"{Path(sys.argv[1]).name}: {len(dets)} cara(s)")
        for box, emb in dets:
            print("  box:", box, "| dims:", emb.shape[0], "| norma:", float(np.linalg.norm(emb)))
