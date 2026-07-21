"""Backup automático con fecha de los .json críticos, ANTES de pisarlos.

Red de seguridad. En un mismo día hubo dos pérdidas de datos por saves que
borraban todo sin querer: la purga automática de huérfanos se comió las
etiquetas, y guardar Ajustes borró la ruta de la biblioteca. `data/` no está en
git (está en .gitignore), así que no había ningún paso atrás.

La idea: cada vez que se va a sobrescribir `tags.json` o `config.json`, primero se
copia el archivo ACTUAL (el bueno, el de antes de este guardado) a
data/backups/ con fecha. Si el guardado nuevo resulta destructivo, la copia
anterior queda a salvo. Se conservan las últimas N por archivo.

Restaurar es a mano y simple: copiar data/backups/tags.AAAAMMDD-HHMMSS.json
sobre data/tags.json (con la app cerrada).
"""

import shutil
import time
from datetime import datetime
from pathlib import Path

from app import DATA_DIR

BACKUPS_DIR = DATA_DIR / "backups"

# Cuántas copias guardar por archivo, y cada cuánto como mucho hacer una nueva.
# El throttle evita que una sesión de etiquetado (muchos guardados seguidos)
# llene la carpeta de copias casi idénticas y rote afuera las útiles: con 120s,
# como mucho una copia cada 2 minutos por archivo.
KEEP = 40
THROTTLE_S = 120


def respaldar(path, keep=KEEP, throttle=THROTTLE_S):
    """Copia `path` (si existe) a data/backups/ con fecha, antes de pisarlo.
    Nunca levanta: un backup que falla NO debe romper el guardado real."""
    try:
        path = Path(path)
        if not path.is_file():
            return                      # primera vez: nada que respaldar
        BACKUPS_DIR.mkdir(parents=True, exist_ok=True)
        patron = f"{path.stem}.*{path.suffix}"
        previas = sorted(BACKUPS_DIR.glob(patron))
        # throttle: si la copia más nueva de ESTE archivo es muy reciente, no
        # duplicar (igual esa copia tiene datos casi actuales)
        if previas:
            try:
                if time.time() - previas[-1].stat().st_mtime < throttle:
                    return
            except OSError:
                pass
        # microsegundos: dos backups en el mismo segundo no colisionan
        ts = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        shutil.copy2(str(path), str(BACKUPS_DIR / f"{path.stem}.{ts}{path.suffix}"))
        # rotar: dejar solo las últimas `keep` de este archivo
        previas = sorted(BACKUPS_DIR.glob(patron))
        for viejo in previas[:-keep]:
            viejo.unlink(missing_ok=True)
    except Exception:
        pass


def listar(stem):
    """Las copias de un archivo ('tags' o 'config'), de la más nueva a la más
    vieja: [(nombre, timestamp_texto)]. Para mostrarlas o restaurar."""
    if not BACKUPS_DIR.is_dir():
        return []
    out = []
    for p in sorted(BACKUPS_DIR.glob(f"{stem}.*"), reverse=True):
        out.append(p.name)
    return out
