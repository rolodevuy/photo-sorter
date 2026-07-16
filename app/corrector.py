"""Corrector de entrenamiento: revisa la carpeta de una persona conocida y
reconstruye sus ejemplos solo con caras verificadas.

Agrupa todas las caras que hay en la carpeta y busca el "grupo dominante": la
persona aparece en su propia carpeta más que nadie. Con eso clasifica:

- OBVIAS: fotos con una sola cara que encaja en el grupo dominante. Se
  confirman solas, sin molestar.
- DUDOSAS: fotos con varias caras (la persona acompañada), o cuya única cara
  desentona con el dominante (posible foto de otra persona). El usuario señala
  cuál es su persona, o "no está".

Al aplicar, la base de esa persona se REEMPLAZA por las caras verificadas:
la contaminación vieja (vectores de intrusos) desaparece. Las fotos donde la
persona no estaba se pueden mover a <carpeta>/_revisar.
"""

import shutil
from pathlib import Path

import numpy as np

from app import CORRECT_THUMBS, IMAGE_EXTENSIONS, REVIEW_DIR_NAME
from app.analyze import _agglomerate, _load_rgb
from app.facedet import detect_and_encode
from app.finder import _save_thumb
from app.known import load_known, save_known

# Umbral para agrupar entre sí las caras de la carpeta (estricto, como el
# re-separado anti-bloques del análisis).
DOM_EPS = 0.45


def scan_person_folder(folder, progress=None):
    """Escanea la carpeta de una persona y clasifica sus fotos.

    Devuelve {"auto": [{rel, enc, thumb}], "review": [{rel, faces: [{j, enc,
    thumb, sug}]}], "no_face": [rel], "n_photos"}. Los recortes van a
    data/correct_thumbs/. Lanza ValueError sin imágenes o sin caras.
    """
    folder = Path(folder)
    photos = [p for p in sorted(folder.rglob("*"))
              if p.suffix.lower() in IMAGE_EXTENSIONS
              and REVIEW_DIR_NAME not in p.parts]
    if not photos:
        raise ValueError(f"No hay imágenes en {folder}")

    if CORRECT_THUMBS.is_dir():
        shutil.rmtree(CORRECT_THUMBS, ignore_errors=True)
    CORRECT_THUMBS.mkdir(parents=True, exist_ok=True)

    entries = []   # {"rel", "faces": [{"j", "enc", "thumb"}]}
    for i, path in enumerate(photos, 1):
        rel = path.relative_to(folder).as_posix()
        if progress:
            progress(i, len(photos), rel)
        try:
            img = _load_rgb(path)
            dets = detect_and_encode(img)
        except Exception:
            continue   # ilegible: fuera del entrenamiento, sin molestar
        faces = []
        for j, (box, enc) in enumerate(dets):
            thumb = f"{i - 1}_{j}.jpg"
            _save_thumb(img, box, CORRECT_THUMBS / thumb)
            faces.append({"j": j, "enc": enc, "thumb": thumb})
        entries.append({"rel": rel, "faces": faces})

    all_encs = [f["enc"] for e in entries for f in e["faces"]]
    if not all_encs:
        raise ValueError("No se detectó ninguna cara en la carpeta.")

    # grupo dominante = el que aparece en más FOTOS (no más caras: una foto
    # ráfaga con 10 caras del mismo intruso no debe ganarle a la persona)
    labels = _agglomerate(np.array(all_encs, dtype=np.float32), DOM_EPS)
    k = 0
    photos_per_label = {}
    for e in entries:
        seen = set()
        for f in e["faces"]:
            f["label"] = int(labels[k])
            k += 1
            seen.add(f["label"])
        for lab in seen:
            photos_per_label[lab] = photos_per_label.get(lab, 0) + 1
    dom = max(photos_per_label, key=photos_per_label.get)

    auto, review, no_face = [], [], []
    for e in entries:
        if not e["faces"]:
            no_face.append(e["rel"])
            continue
        in_dom = [f for f in e["faces"] if f["label"] == dom]
        if len(e["faces"]) == 1 and in_dom:
            f = e["faces"][0]
            auto.append({"rel": e["rel"], "enc": f["enc"], "thumb": f["thumb"]})
        else:
            for f in e["faces"]:
                f["sug"] = f["label"] == dom
            review.append({"rel": e["rel"], "faces": e["faces"]})
    return {"auto": auto, "review": review, "no_face": no_face,
            "n_photos": len(photos)}


def apply_correction(name, folder, auto, review, answers,
                     move_out=True, max_per_person=60):
    """Aplica la corrección: reemplaza los ejemplos de la persona por las
    caras verificadas y (opcional) mueve a _revisar las fotos donde no está.

    answers: {rel: índice_de_cara (str) | "skip" | "none"}.
    - índice: esa cara entrena.
    - "skip": la persona está en la foto pero su cara no salió en los recortes
      (de espaldas, tapada): no entrena y el archivo queda donde está.
    - "none": la persona NO está: no entrena y (con move_out) se aparta.
    Los rel sin respuesta equivalen a "skip". Devuelve (ejemplos, movidas,
    errores). Lanza ValueError si no queda ninguna cara confirmada.
    """
    encs = [a["enc"] for a in auto]
    none_rels = []
    for r in review:
        ans = answers.get(r["rel"])
        if ans is None or ans in ("", "skip"):
            continue
        if ans == "none":
            none_rels.append(r["rel"])
            continue
        face = next((f for f in r["faces"] if str(f["j"]) == str(ans)), None)
        if face:
            encs.append(face["enc"])

    if not encs:
        raise ValueError("No quedó ninguna cara confirmada; la base no se tocó.")

    arr = np.asarray(encs, dtype=np.float32)
    if len(arr) > max_per_person:
        idx = np.linspace(0, len(arr) - 1, max_per_person).astype(int)
        arr = arr[idx]
    db = load_known()
    db[name] = arr   # REEMPLAZA: la contaminación vieja desaparece
    save_known(db)

    moved, errors = [], []
    if move_out and none_rels:
        out = Path(folder) / REVIEW_DIR_NAME
        out.mkdir(parents=True, exist_ok=True)
        for rel in none_rels:
            src = Path(folder) / rel
            if not src.is_file():
                continue
            dest = out / src.name
            n = 2
            while dest.exists():
                dest = out / f"{src.stem}_{n}{src.suffix}"
                n += 1
            try:
                shutil.move(str(src), str(dest))
                moved.append(rel)
            except OSError as e:
                errors.append((rel, str(e)))
    return len(arr), moved, errors
