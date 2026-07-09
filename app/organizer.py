"""Organizer: copia las fotos a carpetas por persona.

Lee los grupos de app.analyze y los nombres puestos en la web, y copia cada
foto a data/sorted/<nombre>/. Una foto con varias personas queda copiada en
la carpeta de cada una. Los originales en data/photos/ no se tocan.

Uso:
    python -m app.organizer
(o desde el botón "Organizar fotos" de la web)
"""

import json
import shutil
import sys

from app import CLUSTERS_PATH, LABELS_PATH, PHOTOS_DIR, SORTED_DIR


def organize():
    """Copia las fotos de los grupos con nombre. Devuelve (copias, personas)."""
    with open(CLUSTERS_PATH, encoding="utf-8") as f:
        data = json.load(f)
    labels = {}
    if LABELS_PATH.is_file():
        with open(LABELS_PATH, encoding="utf-8") as f:
            labels = json.load(f)

    faces = data["faces"]
    copied = 0
    people = set()

    for cluster in data["clusters"]:
        name = labels.get(str(cluster["id"]), "").strip()
        if not name:
            continue

        # caracteres no válidos para nombre de carpeta en Windows
        safe_name = "".join(ch for ch in name if ch not in '<>:"/\\|?*').strip()
        dest_dir = SORTED_DIR / safe_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        people.add(safe_name)

        photos = {faces[fid]["photo"] for fid in cluster["faces"]}
        for rel in sorted(photos):
            src = PHOTOS_DIR / rel
            # aplanar subcarpetas: vacaciones/img.jpg -> vacaciones_img.jpg
            dest = dest_dir / rel.replace("/", "_")
            if src.is_file() and not dest.is_file():
                shutil.copy2(src, dest)
                copied += 1

    return copied, len(people)


def main():
    if not CLUSTERS_PATH.is_file():
        print("[organizer] No existe data/clusters.json. Corré primero: python -m app.analyze")
        sys.exit(1)
    if not LABELS_PATH.is_file():
        print("[organizer] Todavía no hay nombres guardados. Corré la web: python -m app.flask_app")
        sys.exit(1)

    copied, people = organize()
    print(f"[organizer] Listo: {copied} foto(s) copiadas a {SORTED_DIR} "
          f"en {people} carpeta(s) de persona.")


if __name__ == "__main__":
    main()
