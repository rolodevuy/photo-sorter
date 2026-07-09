"""Nomenclador: renombra en masa las fotos de una carpeta con un patrón.

Estructura: <prefijo><separador><número con N dígitos>.<extensión original>
Ejemplos:
    prefijo="vacaciones", separador="_", dígitos=4  ->  vacaciones_0001.jpg
    prefijo="IMG",        separador="",  dígitos=3  ->  IMG001.jpg

Trabaja sobre el primer nivel de la carpeta (no entra en subcarpetas) y solo
sobre imágenes. Renombra en dos pasos (nombre temporal y después el final) para
que nunca se pisen dos archivos entre sí.

Uso normal: desde la web (pestaña "Renombrar").
"""

from pathlib import Path

from app import IMAGE_EXTENSIONS

INVALID = '<>:"/\\|?*'


def _clean_prefix(prefix):
    return "".join(ch for ch in prefix if ch not in INVALID)


def list_targets(folder, order="name"):
    """Imágenes del primer nivel de la carpeta, en el orden elegido."""
    folder = Path(folder)
    files = [p for p in folder.iterdir()
             if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    if order == "date":
        files.sort(key=lambda p: (p.stat().st_mtime, p.name.lower()))
    else:
        files.sort(key=lambda p: p.name.lower())
    return files


def plan_rename(folder, prefix, digits, separator="_", start=1, order="name"):
    """Arma la lista de (nombre_actual, nombre_nuevo) sin tocar nada."""
    prefix = _clean_prefix(prefix)
    digits = max(1, min(int(digits), 12))
    files = list_targets(folder, order)
    plan = []
    for i, path in enumerate(files):
        num = str(start + i).zfill(digits)
        new_name = f"{prefix}{separator}{num}{path.suffix.lower()}"
        plan.append((path.name, new_name))
    return plan


def apply_rename(folder, prefix, digits, separator="_", start=1, order="name"):
    """Renombra en dos fases (evita choques). Devuelve (renombrados, errores)."""
    folder = Path(folder)
    prefix = _clean_prefix(prefix)
    digits = max(1, min(int(digits), 12))
    files = list_targets(folder, order)

    renamed = 0
    errors = []
    temp_pairs = []  # (ruta_temporal, nombre_final)

    # Fase 1: a nombre temporal único (así los nombres finales no chocan con
    # los actuales, aunque el patrón nuevo coincida con nombres existentes).
    for i, path in enumerate(files):
        num = str(start + i).zfill(digits)
        final_name = f"{prefix}{separator}{num}{path.suffix.lower()}"
        tmp = folder / f".__rename_tmp_{i:06d}{path.suffix.lower()}"
        try:
            path.rename(tmp)
            temp_pairs.append((tmp, final_name))
        except OSError as e:
            errors.append((path.name, str(e)))

    # Fase 2: del temporal al nombre final.
    for tmp, final_name in temp_pairs:
        dest = folder / final_name
        try:
            if dest.exists():
                # otro archivo (no de esta tanda) ya tiene ese nombre
                errors.append((final_name, "ya existe un archivo con ese nombre"))
                tmp.rename(folder / f"{tmp.stem}_sin_renombrar{tmp.suffix}")
                continue
            tmp.rename(dest)
            renamed += 1
        except OSError as e:
            errors.append((final_name, str(e)))

    return renamed, errors
