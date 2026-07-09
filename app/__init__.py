"""photo-sorter: agrupa fotos por rostros parecidos, 100% local."""

from pathlib import Path

# Rutas internas del proyecto (solo datos generados; las fotos del usuario
# quedan donde estén: origen y destino se eligen desde la web).
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
FACES_DIR = DATA_DIR / "faces"          # miniaturas de rostros recortados
CLUSTERS_PATH = DATA_DIR / "clusters.json"
LABELS_PATH = DATA_DIR / "labels.json"
CONFIG_PATH = DATA_DIR / "config.json"  # carpetas de origen y destino elegidas

# Extensiones de imagen soportadas
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
