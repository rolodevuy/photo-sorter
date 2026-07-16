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
KNOWN_THUMBS = DATA_DIR / "known_thumbs"    # miniatura representativa de cada persona conocida
NOT_FACES_PATH = DATA_DIR / "not_faces.json"  # lista negra: detecciones que NO son personas
STATS_PATH = DATA_DIR / "stats.json"        # aciertos de las sugerencias (calidad del reconocimiento)
FIND_THUMBS = DATA_DIR / "find_thumbs"      # miniaturas de la última búsqueda de persona
PICK_THUMBS = DATA_DIR / "pick_thumbs"      # recortes del selector "¿cuál es la cara?" (portadas)
CORRECT_THUMBS = DATA_DIR / "correct_thumbs"  # recortes del corrector de entrenamiento
PERSON_META_PATH = DATA_DIR / "person_meta.json"  # última corrección / búsqueda por persona
TAGS_PATH = DATA_DIR / "tags.json"          # etiquetas manuales: por persona y por foto
GALLERY_THUMBS = DATA_DIR / "gallery_thumbs"  # miniaturas cacheadas de las galerías (rápido)

# Subcarpeta (dentro de la carpeta de una persona) a donde el corrector mueve
# las fotos donde la persona NO está. Se excluye de galerías y escaneos.
REVIEW_DIR_NAME = "_revisar"

# Subcarpeta a donde el revisor NSFW aparta las fotos marcadas. En la galería
# se ocultan por defecto y se muestran con un botón.
NSFW_DIR_NAME = "_nsfw"

# Subcarpeta a donde el separador manda las fotos con MÁS DE UNA cara. El
# importador las saltea: así solo aprende de fotos donde la persona está sola
# (ahí "la cara más grande" es infalible) y las grupales se procesan aparte
# con el corrector, que pregunta cuál es cuál.
GROUP_DIR_NAME = "_grupo"

# Extensiones de imagen soportadas por el ANÁLISIS de rostros.
# Los GIF se analizan por su primer fotograma (si están animados, se ignora
# el resto). Los archivos con extensión "mentida" (ej: un PNG llamado .jpg)
# igual se leen bien porque la librería detecta el formato por el contenido.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}

# Videos que también entran al ANÁLISIS de rostros: se toman fotogramas
# (1 por segundo) y se buscan caras igual que en una foto. Solo los formatos
# que OpenCV lee bien en Windows; si uno no se puede abrir, se saltea con
# aviso como una imagen rota.
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}

# Formatos que puede tocar el NOMENCLADOR (renombrar). Es solo renombrado por
# patrón: no se lee ni analiza el contenido. Cubre imágenes y videos comunes.
RENAME_EXTENSIONS = IMAGE_EXTENSIONS | {
    ".jfif", ".tif", ".tiff", ".heic", ".heif", ".avif", ".ico",  # imágenes
    ".mp4", ".mov", ".avi", ".webm", ".mkv", ".m4v", ".wmv",       # video
    ".flv", ".mpg", ".mpeg", ".m2ts", ".mts", ".3gp",              # video
}
