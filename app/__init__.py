"""photo-sorter: organizador de fotos por reconocimiento facial."""

from pathlib import Path

# Rutas base del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
TRAINING_DIR = DATA_DIR / "training"
PHOTOS_DIR = DATA_DIR / "photos"
ENCODINGS_PATH = DATA_DIR / "encodings.pkl"
DETECTIONS_PATH = DATA_DIR / "detections.json"

# Extensiones de imagen soportadas
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
