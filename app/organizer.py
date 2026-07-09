"""Organizer: copia las fotos a carpetas por persona en la carpeta de destino.

Lee los grupos de app.analyze y los nombres puestos en la web, y copia cada
foto a <destino>/<nombre>/. Una foto con varias personas queda copiada en la
carpeta de cada una. Los originales no se tocan.

Uso normal: botón "Organizar fotos" de la web.
También por consola: python -m app.organizer
"""

import json
import shutil
import sys
from pathlib import Path

from app import CLUSTERS_PATH, LABELS_PATH
from app.config import load_config


def organize():
    """Copia las fotos de los grupos con nombre. Devuelve (copias, personas, destino)."""
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
    faces = data["faces"]
    copied = 0
    people = set()

    for cluster in data["clusters"]:
        name = labels.get(str(cluster["id"]), "").strip()
        if not name:
            continue

        # caracteres no válidos para nombre de carpeta en Windows
        safe_name = "".join(ch for ch in name if ch not in '<>:"/\\|?*').strip()
        dest_dir = output_dir / safe_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        people.add(safe_name)

        photos = {faces[fid]["photo"] for fid in cluster["faces"]}
        for rel in sorted(photos):
            src = photos_dir / rel
            # aplanar subcarpetas: vacaciones/img.jpg -> vacaciones_img.jpg
            dest = dest_dir / rel.replace("/", "_")
            if src.is_file() and not dest.is_file():
                shutil.copy2(src, dest)
                copied += 1

    return copied, len(people), output_dir


def main():
    if not CLUSTERS_PATH.is_file():
        print("[organizer] No hay análisis todavía. Corré la web: python -m app.flask_app")
        sys.exit(1)

    try:
        copied, people, output_dir = organize()
    except ValueError as e:
        print(f"[organizer] {e}")
        sys.exit(1)

    print(f"[organizer] Listo: {copied} foto(s) copiadas a {output_dir} "
          f"en {people} carpeta(s) de persona.")


if __name__ == "__main__":
    main()
