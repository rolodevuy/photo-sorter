"""photo-sorter: agrupa fotos por rostros parecidos, 100% local."""

from pathlib import Path

# Rutas base del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PHOTOS_DIR = DATA_DIR / "photos"        # fotos originales a clasificar
FACES_DIR = DATA_DIR / "faces"          # miniaturas de rostros recortados
SORTED_DIR = DATA_DIR / "sorted"        # salida: carpetas por persona
CLUSTERS_PATH = DATA_DIR / "clusters.json"
LABELS_PATH = DATA_DIR / "labels.json"

# Extensiones de imagen soportadas
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
