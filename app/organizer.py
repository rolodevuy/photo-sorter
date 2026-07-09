"""Organizer: mueve las fotos a carpetas por persona en la carpeta de destino.

Lee los grupos de app.analyze y los nombres puestos en la web, y MUEVE cada
foto a <destino>/<nombre>/ como <nombre>_0000.jpg. La foto desaparece del
origen (es para ordenar, no para duplicar).

Caso especial: una foto donde aparecen varias personas con nombre se copia a
la carpeta de cada una y el original se borra recién al final (así queda en
todas sin perderse). Si una foto no se puede mover (permiso, archivo abierto),
se deja en el origen y se informa.

Uso normal: botón "Organizar fotos" de la web.
También por consola: python -m app.organizer
"""

import json
import shutil
import sys
from pathlib import Path

from app import CLUSTERS_PATH, LABELS_PATH
from app.config import load_config

FILE_ATTRIBUTE_HIDDEN = 0x2


def _unhide(path):
    """Saca el atributo 'oculto' de Windows (heredado del original al copiar)."""
    import ctypes
    attrs = ctypes.windll.kernel32.GetFileAttributesW(str(path))
    if attrs != -1 and attrs & FILE_ATTRIBUTE_HIDDEN:
        ctypes.windll.kernel32.SetFileAttributesW(str(path), attrs & ~FILE_ATTRIBUTE_HIDDEN)


def _plan(data, labels, output_dir):
    """Arma el plan de movimientos: {ruta_origen: [destinos]}.

    Recorre los grupos con nombre y asigna a cada foto sus carpetas de destino
    (una foto con varias personas tiene más de un destino). Los nombres de
    salida son <nombre>_0000, _0001... por persona.
    """
    faces = data["faces"]
    plan = {}          # rel_origen -> [Path destino, ...]
    people = set()

    for cluster in data["clusters"]:
        name = labels.get(str(cluster["id"]), "").strip()
        if not name:
            continue

        # caracteres no válidos para nombre de carpeta en Windows
        safe_name = "".join(ch for ch in name if ch not in '<>:"/\\|?*').strip()
        if not safe_name:
            continue
        dest_dir = output_dir / safe_name
        people.add(safe_name)

        photos = sorted({faces[fid]["photo"] for fid in cluster["faces"]})
        for i, rel in enumerate(photos):
            dest = dest_dir / f"{safe_name}_{i:04d}{Path(rel).suffix.lower()}"
            plan.setdefault(rel, []).append(dest)

    return plan, people


def organize():
    """Mueve las fotos según los grupos con nombre.

    Devuelve (movidas, personas, destino, errores) donde `errores` es una lista
    de (ruta, motivo) de las fotos que no se pudieron mover.
    """
    cfg = load_config()
    if not cfg.get("output_dir"):
        raise ValueError("Todavía no elegiste la carpeta de destino.")
    output_dir = Path(cfg["output_dir"])

    with open(CLUSTERS_PATH, encoding="utf-8") as f:
        data = json.load(f)
    labels = {}
    if LABELS_PATH.is_file():
        with open(LABELS_PATH, encoding="utf-8") as f:
            labels = json.load(f)

    photos_dir = Path(data["photos_dir"])
    plan, people = _plan(data, labels, output_dir)

    moved = 0
    errors = []

    for rel, dests in plan.items():
        src = photos_dir / rel
        if not src.is_file():
            continue
        try:
            # Copiar a cada destino (creando carpetas). Con una sola persona
            # es una sola copia; con varias, una por carpeta.
            for dest in dests:
                dest.parent.mkdir(parents=True, exist_ok=True)
                if not dest.is_file():
                    shutil.copy2(src, dest)
                    _unhide(dest)
            # Recién cuando todas las copias salieron bien, borrar el original.
            src.unlink()
            moved += 1
        except OSError as e:
            # No se pudo mover: dejar el original donde está y avisar.
            errors.append((rel, str(e)))

    return moved, len(people), output_dir, errors


def main():
    if not CLUSTERS_PATH.is_file():
        print("[organizer] No hay análisis todavía. Corré la web: python -m app.flask_app")
        sys.exit(1)

    try:
        moved, people, output_dir, errors = organize()
    except ValueError as e:
        print(f"[organizer] {e}")
        sys.exit(1)

    print(f"[organizer] Listo: {moved} foto(s) movidas a {output_dir} "
          f"en {people} carpeta(s) de persona.")
    if errors:
        print(f"[organizer] {len(errors)} foto(s) no se pudieron mover (quedaron en el origen):")
        for rel, reason in errors[:10]:
            print(f"  - {rel}: {reason}")


if __name__ == "__main__":
    main()
