"""Clasificador de ACTO/posición sexual en una foto (porntech/sex-position).

Un resnet50 de timm que en UNA pasada (rápida, en CPU) dice qué acto hay en una
imagen NSFW. Tapa el punto débil del VLM, que el acto lo adivina; acá es una
respuesta anclada de un modelo dedicado.

OJO — solo tiene sentido en fotos NSFW: con gente vestida o normal tira cualquier
cosa (el propio autor lo avisa). El llamador lo corre SOLO en fotos marcadas NSFW
y NUNCA en personas etiquetadas como menor.

Todo local: baja los pesos de HuggingFace una vez (a la caché de HF_HOME, que
apunta a E:) y después corre sin conexión. La foto no sale de la máquina.

Ojo con "después corre sin conexión": no era gratis. timm con pretrained=True
consulta el Hub CADA vez que carga el modelo en frío, aunque los pesos ya estén
en la caché — solo para ver si hay versión nueva. De ahí salía el aviso de
"unauthenticated requests to the HF Hub". No mandaba ninguna foto, pero era
tráfico a internet que no hacía falta. Abajo se apaga cuando ya está cacheado.
"""

import os
import threading
from pathlib import Path

from PIL import Image

# Si los pesos YA están en la caché, modo offline: timm los usa del disco y no
# toca la red. Si no están (primera vez), se deja online para poder bajarlos.
# Va acá arriba y no dentro de _cargar() a propósito: huggingface_hub lee esta
# variable UNA vez, cuando se importa, así que cambiarla más tarde no sirve de
# nada. timm se importa perezosamente más abajo, así que todavía llegamos.
_CACHE_HF = Path(os.environ.get("HF_HOME") or (Path.home() / ".cache" / "huggingface"))
if (_CACHE_HF / "hub" / "models--porntech--sex-position").is_dir():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

# Orden EXACTO de las clases del modelo (los índices de salida son estos):
LABELS = ["blowjob", "hardcore", "titjob", "handjob",
          "pussy-licking", "fingering", "other", "solo"]

# A la jerga rioplatense. "other" se descarta (no es una etiqueta útil).
# "hardcore" es penetración genérica (el modelo no separa misionero/doggy/etc.).
_MAP = {
    "blowjob": "chupando",
    "hardcore": "penetración",
    "titjob": "rusa",
    "handjob": "paja",
    "pussy-licking": "comiéndole la concha",
    "fingering": "dedeando",
    "solo": "sola",
}

UMBRAL = 0.45          # confianza mínima para sugerir el acto

_MODEL = None
_TRANSFORM = None
_LOCK = threading.Lock()


def disponible():
    """True si timm está instalado (el modelo se baja solo la primera vez)."""
    try:
        import timm  # noqa: F401
        return True
    except Exception:
        return False


def _cargar():
    """Carga perezosa: la primera vez baja pesos y arma el modelo + transform."""
    global _MODEL, _TRANSFORM
    if _MODEL is not None:
        return
    import timm
    from timm.data import create_transform, resolve_data_config
    m = timm.create_model("hf_hub:porntech/sex-position", pretrained=True)
    m.eval()
    _TRANSFORM = create_transform(**resolve_data_config({}, model=m))
    _MODEL = m


def clasificar(image_path):
    """El acto detectado en la foto, en jerga: ("chupando", 0.83), o None si el
    modelo no está seguro (< UMBRAL) o dice 'other'. Un solo forward en CPU.

    Lanza si timm/el modelo fallan; el llamador lo envuelve en try."""
    import torch
    with _LOCK:
        _cargar()
        img = Image.open(image_path).convert("RGB")
        x = _TRANSFORM(img).unsqueeze(0)
        with torch.no_grad():
            probs = _MODEL(x).softmax(1)[0]
        i = int(probs.argmax())
        conf = float(probs[i])
        etq = LABELS[i]
    if etq == "other" or conf < UMBRAL:
        return None
    return (_MAP.get(etq, etq), conf)
