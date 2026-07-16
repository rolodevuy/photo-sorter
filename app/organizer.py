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

from app import CLUSTERS_PATH, FACES_DIR, LABELS_PATH
from app.config import get_library, load_config

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
    library = get_library(cfg)
    if not library:
        raise ValueError("Todavía no elegiste la carpeta de la Biblioteca (pestaña Ajustes).")
    output_dir = Path(library)

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

    # Los grupos organizados ya cumplieron: se sacan de la sesión para que la
    # web no muestre para siempre el último análisis. Quedan los grupos sin
    # nombre (por si querés seguir nombrando) y los que tuvieron errores.
    if moved:
        _prune_organized(data, labels, {rel for rel, _ in errors})

    return moved, len(people), output_dir, errors


def _prune_organized(data, labels, failed):
    """Saca de clusters.json los grupos con nombre cuyas fotos ya se movieron.
    Si alguna foto del grupo falló, el grupo se queda para poder reintentar."""
    faces = data["faces"]
    keep, removed_faces = [], []
    for cluster in data["clusters"]:
        name = labels.get(str(cluster["id"]), "").strip()
        safe = "".join(ch for ch in name if ch not in '<>:"/\\|?*').strip()
        photos = {faces[fid]["photo"] for fid in cluster["faces"]}
        if safe and not (photos & failed):
            removed_faces.extend(cluster["faces"])
            labels.pop(str(cluster["id"]), None)
        else:
            keep.append(cluster)

    if len(keep) == len(data["clusters"]):
        return
    data["clusters"] = keep

    # limpiar las caras (y miniaturas) que solo estaban en los grupos sacados
    kept_fids = {fid for c in keep for fid in c["faces"]}
    for fid in removed_faces:
        if fid not in kept_fids:
            faces.pop(fid, None)
            (FACES_DIR / f"{fid}.jpg").unlink(missing_ok=True)

    with open(CLUSTERS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)


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
