"""Buscador de fotos duplicadas o casi iguales, 100% local.

Detecta dos tipos de repetidos en una carpeta:
- EXACTOS: el mismo archivo aunque tenga distinto nombre (mismo contenido byte
  a byte, comparado por hash SHA-1).
- PARECIDOS: la misma imagen reescalada, recomprimida, con marca de agua o
  guardada dos veces. Se comparan con un "hash perceptual" (dHash): dos fotos
  visualmente iguales dan hashes casi idénticos.

Genera data/duplicates.json con los grupos y miniaturas en data/dupe_thumbs/.
Dentro de cada grupo se sugiere cuál conservar (la de mayor resolución y peso).

Uso normal: desde la web (pestaña "Buscar duplicados").
También por consola (usa la carpeta de origen guardada): python -m app.duplicates
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from app import DUPES_PATH, DUPES_THUMBS, IMAGE_EXTENSIONS
from app.config import load_config

# Distancia de Hamming máxima entre hashes perceptuales para considerar dos
# fotos "parecidas" (0 = idénticas). Sobre 64 bits, hasta ~5 es muy parecido.
DEFAULT_THRESHOLD = 5

THUMBNAIL_SIZE = 200
HASH_SIZE = 8  # dHash de 8x8 -> 64 bits


def dhash(image):
    """Hash perceptual (difference hash) de 64 bits a partir de una imagen PIL."""
    small = image.convert("L").resize((HASH_SIZE + 1, HASH_SIZE), Image.LANCZOS)
    px = np.asarray(small, dtype=np.int16)
    # compara cada pixel con el de su derecha
    diff = px[:, 1:] > px[:, :-1]
    bits = 0
    for b in diff.flatten():
        bits = (bits << 1) | int(b)
    return bits


def file_hash(path, chunk=1 << 20):
    """SHA-1 del contenido del archivo (para duplicados exactos)."""
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


class _UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


DUPES_FOLDER_NAME = "_duplicados"


def _iter_images(folder, exclude=None):
    for p in sorted(folder.rglob("*")):
        if exclude and p.is_relative_to(exclude):
            continue
        # nunca re-escanear la carpeta donde movemos los sobrantes
        if DUPES_FOLDER_NAME in p.parts:
            continue
        if p.suffix.lower() in IMAGE_EXTENSIONS:
            yield p


def find_duplicates(folder, threshold=DEFAULT_THRESHOLD, exclude=None, progress=None, log=print):
    """Busca duplicados exactos y parecidos en `folder`.

    Devuelve la cantidad de grupos encontrados. Escribe data/duplicates.json.
    progress: callback opcional progress(actual, total, nombre).
    """
    folder = Path(folder)
    paths = list(_iter_images(folder, exclude))
    if not paths:
        raise ValueError(f"No hay imágenes en {folder}")

    if DUPES_THUMBS.is_dir():
        for old in DUPES_THUMBS.glob("*.jpg"):
            old.unlink()
    DUPES_THUMBS.mkdir(parents=True, exist_ok=True)

    items = []  # {"rel", "size", "w", "h", "sha", "phash"}
    hashes = []
    for i, path in enumerate(paths):
        if progress:
            progress(i + 1, len(paths), path.name)
        try:
            with Image.open(path) as img:
                w, h = img.size
                ph = dhash(img)
                thumb = img.convert("RGB")
                thumb.thumbnail((THUMBNAIL_SIZE, THUMBNAIL_SIZE))
        except Exception as e:
            log(f"[dupes] no se pudo leer {path.name}: {e}")
            continue

        idx = len(items)
        thumb.save(DUPES_THUMBS / f"{idx:06d}.jpg", quality=80)
        items.append({
            "rel": path.relative_to(folder).as_posix(),
            "size": path.stat().st_size,
            "w": w, "h": h,
            "sha": file_hash(path),
        })
        hashes.append(ph)

    if not items:
        raise ValueError("No se pudo leer ninguna imagen.")

    uf = _UnionFind(len(items))

    # 1) Duplicados exactos: mismo SHA-1
    by_sha = {}
    for i, it in enumerate(items):
        by_sha.setdefault(it["sha"], []).append(i)
    for group in by_sha.values():
        for j in group[1:]:
            uf.union(group[0], j)

    # 2) Parecidos: distancia de Hamming <= threshold (vectorizado con numpy)
    harr = np.array(hashes, dtype=np.uint64)
    for i in range(len(items)):
        dist = np.bitwise_count(harr ^ harr[i])
        for j in np.nonzero(dist <= threshold)[0]:
            if j > i:
                uf.union(i, int(j))

    # Armar grupos (solo los de 2 o más)
    comps = {}
    for i in range(len(items)):
        comps.setdefault(uf.find(i), []).append(i)

    groups = []
    for members in comps.values():
        if len(members) < 2:
            continue
        # keeper: mayor resolución, y a igualdad, mayor peso
        keeper = max(members, key=lambda i: (items[i]["w"] * items[i]["h"], items[i]["size"]))
        keeper_sha = items[keeper]["sha"]
        files = []
        for i in sorted(members, key=lambda i: (items[i]["w"] * items[i]["h"], items[i]["size"]), reverse=True):
            files.append({
                "idx": i,
                "rel": items[i]["rel"],
                "size": items[i]["size"],
                "w": items[i]["w"], "h": items[i]["h"],
                "keeper": i == keeper,
                "exact": items[i]["sha"] == keeper_sha,
            })
        groups.append({
            "files": files,
            "exact": all(f["exact"] for f in files),
        })

    # grupos más grandes primero
    groups.sort(key=lambda g: len(g["files"]), reverse=True)

    with open(DUPES_PATH, "w", encoding="utf-8") as f:
        json.dump({"folder": str(folder), "threshold": threshold, "groups": groups},
                  f, ensure_ascii=False, indent=2)

    total_dupes = sum(len(g["files"]) - 1 for g in groups)
    log(f"[dupes] {len(groups)} grupo(s) de duplicados; {total_dupes} foto(s) sobrantes.")
    return len(groups)


def resolve_duplicates(rels):
    """Mueve las fotos indicadas (rutas relativas al origen) a <origen>/_duplicados/.

    No borra nada: solo las aparta para que puedas revisarlas antes de eliminarlas.
    Devuelve (movidas, carpeta_duplicados, errores).
    """
    with open(DUPES_PATH, encoding="utf-8") as f:
        data = json.load(f)
    folder = Path(data["folder"])
    dupes_dir = folder / DUPES_FOLDER_NAME
    dupes_dir.mkdir(parents=True, exist_ok=True)

    import shutil
    moved = 0
    errors = []
    for rel in rels:
        src = folder / rel
        if not src.is_file():
            continue
        # aplanar subcarpetas y evitar choques de nombre
        base = rel.replace("/", "_")
        dest = dupes_dir / base
        n = 1
        while dest.exists():
            dest = dupes_dir / f"{Path(base).stem}_{n}{Path(base).suffix}"
            n += 1
        try:
            shutil.move(str(src), str(dest))
            moved += 1
        except OSError as e:
            errors.append((rel, str(e)))
    return moved, dupes_dir, errors


def main():
    cfg = load_config()
    folder = cfg.get("photos_dir")
    if not folder or not Path(folder).is_dir():
        print("[dupes] Elegí la carpeta de origen en la web: python -m app.flask_app")
        sys.exit(1)
    exclude = Path(cfg["output_dir"]) if cfg.get("output_dir") else None
    try:
        find_duplicates(folder, exclude=exclude)
    except ValueError as e:
        print(f"[dupes] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
