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
DUPES_PATH = DATA_DIR / "duplicates.json"   # grupos de fotos duplicadas
DUPES_THUMBS = DATA_DIR / "dupe_thumbs"     # miniaturas para la vista de duplicados
ENCODINGS_NPY = DATA_DIR / "encodings.npy"  # vectores de las caras del último análisis
KNOWN_PATH = DATA_DIR / "known_people.json" # base de personas conocidas (nombre -> vectores)

# Extensiones de imagen soportadas.
# Los GIF se analizan por su primer fotograma (si están animados, se ignora
# el resto). Los archivos con extensión "mentida" (ej: un PNG llamado .jpg)
# igual se leen bien porque la librería detecta el formato por el contenido.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}
