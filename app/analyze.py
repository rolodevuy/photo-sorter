"""Analyze: detecta rostros en la carpeta de origen y agrupa los parecidos.

No necesita entrenamiento previo: recorre la carpeta elegida (los archivos NO
se mueven ni se copian), calcula un encoding por cada rostro encontrado y
agrupa los que pertenecen a la misma persona (clustering con DBSCAN). También
analiza videos: toma un fotograma por segundo y busca caras en cada uno,
quedándose con una sola cara por persona por video. Guarda:

- data/faces/<face_id>.jpg   miniatura de cada rostro (para la web)
- data/clusters.json         rostros + grupos + carpeta de origen usada

Todo corre local, sin ninguna conexión a internet.

Uso normal: desde la web (python -m app.flask_app).
También por consola (usa las carpetas guardadas): python -m app.analyze
"""

import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
from sklearn.cluster import AgglomerativeClustering

from app import (
    CLUSTERS_PATH, ENCODINGS_NPY, FACES_DIR, IMAGE_EXTENSIONS, LABELS_PATH,
    VIDEO_EXTENSIONS,
)
from app.config import load_config
from app.facedet import detect_and_encode
from app.known import identify_ranked, load_known
from app.notfaces import is_not_face, load_not_faces
from app.stats import archive_current

# Umbral de distancia coseno para agrupar caras (agrupado jerárquico con
# enlace "average"). Calibrado con 433 caras reales de 89 personas.
#
# Se usa AgglomerativeClustering (no DBSCAN) porque DBSCAN "encadena": al subir
# el umbral para aunar mejor, un solo enganche malo une dos personas y por
# cadena termina metiendo decenas de personas en un grupo (a 0.60 llegaba a 44).
# El enlace "average" no encadena: aunque suba el umbral, el peor grupo mezcla
# a lo sumo 2-3 caras muy parecidas. A 0.60 aúna bien (una persona con muchas
# fotos queda junta) sin armar bloques.
#
# Más bajo = más estricto (una persona puede partirse en varios grupos; les
# ponés el mismo nombre y se unen al organizar). Más alto = aúna más, con algo
# más de riesgo de mezclar parecidos.
EPS = 0.60

# Segundo paso anti-bloques: después de agrupar, un grupo se considera "difuso"
# (probable mezcla de personas) si sus caras están en promedio lejos de su
# centro; se re-separa con un umbral más estricto. Un grupo grande pero compacto
# (una persona con muchas fotos) tiene cohesión baja y NO se toca.
SPLIT_MIN_SIZE = 4      # solo grupos de este tamaño o más
SPLIT_COHESION = 0.42   # distancia media al centro por encima de la cual se re-separa
SPLIT_EPS = 0.45        # umbral más estricto para re-separar el grupo difuso

# Distancia coseno máxima para SUGERIR una persona conocida en un grupo. Es más
# permisiva que la del reconocimiento estricto (known.MATCH_EPS): acá alcanza
# con que sea el candidato más parecido y razonable, porque el usuario confirma.
SUGGEST_EPS = 0.62

THUMBNAIL_SIZE = 160  # px del lado mayor de la miniatura
BOX_MARGIN = 0.25     # margen extra alrededor del rostro al recortar

# Videos: se toma un fotograma por segundo. Dentro de un mismo video, dos
# caras a menos de esta distancia coseno se consideran la misma persona y se
# guarda una sola (la más grande, que suele ser la más nítida). Es más
# estricto que los umbrales de agrupado (EPS/SPLIT_EPS) para no fundir a dos
# personas parecidas que salen en el mismo video.
FRAME_STEP_S = 1.0
VIDEO_DEDUPE_EPS = 0.35

# Tamaño mínimo del rostro en píxeles (lado menor del recuadro), en fotos y
# videos. Más chico que esto (típico: gente de fondo) la cara sale borrosa y
# su vector es ruidoso: no se deduplica bien y llena el asistente de grupos
# basura de una sola cara. Los modelos alinean a 112px, así que por debajo de
# ~80px la huella pierde calidad. Configurable desde la web (por eso es solo
# el valor por defecto).
MIN_FACE_PX = 80


def _load_rgb(path):
    """Carga una imagen como numpy RGB, respetando la orientación EXIF."""
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    return np.asarray(img)


def list_photos(photos_dir, exclude=None, media="both"):
    """Lista los archivos del origen a analizar, salteando la carpeta de
    destino. media: "images" (solo fotos), "videos" (solo videos) o "both"."""
    exts = {
        "images": IMAGE_EXTENSIONS,
        "videos": VIDEO_EXTENSIONS,
    }.get(media, IMAGE_EXTENSIONS | VIDEO_EXTENSIONS)
    photos = []
    for p in sorted(photos_dir.rglob("*")):
        if exclude and p.is_relative_to(exclude):
            continue
        if p.suffix.lower() in exts:
            photos.append(p)
    return photos


def save_thumbnail(image, box, face_id):
    """Recorta el rostro (con margen) y guarda una miniatura."""
    top, right, bottom, left = box
    h, w = image.shape[:2]
    margin_y = int((bottom - top) * BOX_MARGIN)
    margin_x = int((right - left) * BOX_MARGIN)
    crop = image[
        max(0, top - margin_y):min(h, bottom + margin_y),
        max(0, left - margin_x):min(w, right + margin_x),
    ]
    thumb = Image.fromarray(crop)
    thumb.thumbnail((THUMBNAIL_SIZE, THUMBNAIL_SIZE))
    thumb.save(FACES_DIR / f"{face_id}.jpg", quality=85)


def _box_area(box):
    top, right, bottom, left = box
    return max(0, bottom - top) * max(0, right - left)


def _big_enough(box, min_px=MIN_FACE_PX):
    top, right, bottom, left = box
    return min(bottom - top, right - left) >= min_px


def _keep_best_face(kept, frame, box, enc, t):
    """Dedupe dentro de un video: si la cara ya está (misma persona), se queda
    la de mayor tamaño; si es nueva, se suma a la lista."""
    for k in kept:
        if 1.0 - float(np.dot(k["enc"], enc)) < VIDEO_DEDUPE_EPS:
            if _box_area(box) > _box_area(k["box"]):
                k.update(frame=frame, box=box, enc=enc, t=t)
            return
    kept.append({"frame": frame, "box": box, "enc": enc, "t": t})


def _scan_video(path, should_stop=None, black=None, min_px=MIN_FACE_PX):
    """Muestrea un video (1 fotograma por segundo) y detecta caras.

    Devuelve (kept, muestreados): kept es la lista de caras únicas del video,
    cada una como dict {frame, box, enc, t} con el fotograma donde salió más
    grande. black es la lista negra de no-personas (se descartan solas).
    min_px: tamaño mínimo de cara. Lanza ValueError si el video no se puede abrir.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError("no se pudo abrir el video")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        # sin FPS confiable (algunos contenedores) se asume 30
        step = max(1, round((fps if fps and fps > 0 else 30) * FRAME_STEP_S))
        kept = []
        frame_idx = 0
        sampled = 0
        while True:
            if should_stop and should_stop():
                break
            if not cap.grab():
                break
            if frame_idx % step == 0:
                ok, bgr = cap.retrieve()
                if ok:
                    sampled += 1
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    t = frame_idx / fps if fps and fps > 0 else float(frame_idx)
                    for box, enc in detect_and_encode(rgb):
                        if _big_enough(box, min_px) and not is_not_face(enc, black):
                            _keep_best_face(kept, rgb, box, enc, t)
            frame_idx += 1
    finally:
        cap.release()
    return kept, sampled


def _agglomerate(vectors, eps):
    """Etiquetas de agrupado jerárquico (average) para una matriz de vectores."""
    if len(vectors) == 1:
        return [0]
    return AgglomerativeClustering(
        n_clusters=None, distance_threshold=eps,
        metric="cosine", linkage="average",
    ).fit_predict(vectors)


def _cohesion(vectors):
    """Distancia coseno media de las caras al centro del grupo. Baja = compacto
    (probable una persona); alta = disperso (probable mezcla)."""
    cent = vectors.mean(axis=0)
    n = np.linalg.norm(cent)
    if n > 0:
        cent = cent / n
    return float((1.0 - vectors @ cent).mean())


def cluster_faces(faces, encodings):
    """Agrupa los encodings por persona (jerárquico, enlace average) y re-separa
    los grupos "difusos" (probable mezcla de personas). Devuelve lista de
    clusters; las caras que no se parecen a ninguna otra quedan como grupos de
    una sola cara."""
    face_ids = list(faces.keys())
    X = np.array(encodings)
    labels = _agglomerate(X, EPS)

    # agrupar índices por etiqueta
    by_label = {}
    for idx, label in enumerate(labels):
        by_label.setdefault(int(label), []).append(idx)

    # segundo paso: re-separar los grupos grandes y dispersos
    final_groups = []  # listas de índices
    for members in by_label.values():
        if len(members) >= SPLIT_MIN_SIZE and _cohesion(X[members]) > SPLIT_COHESION:
            sub = X[members]
            sub_labels = _agglomerate(sub, SPLIT_EPS)
            parts = {}
            for local_i, sl in enumerate(sub_labels):
                parts.setdefault(int(sl), []).append(members[local_i])
            final_groups.extend(parts.values())
        else:
            final_groups.append(members)

    # pasar de índices a face_ids, grupos grandes primero
    final_groups.sort(key=len, reverse=True)
    return [
        {"id": i, "faces": [face_ids[idx] for idx in members]}
        for i, members in enumerate(final_groups)
    ]


def run_analysis(photos_dir, exclude=None, progress=None, log=print,
                 should_stop=None, min_face_px=MIN_FACE_PX, media="both"):
    """Analiza la carpeta de origen y escribe clusters.json.

    Usa YuNet (detección) + SFace (reconocimiento) de OpenCV, que detectan bien
    caras anguladas o de perfil y son rápidos en CPU.
    progress: callback opcional progress(actual, total, nombre_foto).
    should_stop: callback opcional que, si devuelve True, corta el análisis y
    guarda lo procesado hasta ese momento (parcial).
    min_face_px: tamaño mínimo de cara; más grande = menos gente de fondo, menos
    grupos basura.
    Devuelve (cantidad_rostros, cantidad_grupos). Lanza ValueError si no hay
    fotos o no se encuentra ningún rostro.
    """
    photos_dir = Path(photos_dir)
    photos = list_photos(photos_dir, exclude, media)
    if not photos:
        que = {"images": "imágenes", "videos": "videos"}.get(media, "imágenes ni videos")
        raise ValueError(f"No hay {que} en {photos_dir}")

    # Miniaturas de corridas anteriores fuera; el análisis rehace todo.
    if FACES_DIR.is_dir():
        shutil.rmtree(FACES_DIR)
    FACES_DIR.mkdir(parents=True, exist_ok=True)

    faces = {}      # face_id -> {"photo": ruta relativa al origen, "box": [t,r,b,l]}
    encodings = []
    no_faces = []
    counter = 0
    stopped = False

    # Lista negra: detecciones ya marcadas como "no es una persona" (tapizados,
    # objetos); las parecidas se descartan sin molestar.
    black = load_not_faces()
    if black is not None:
        log(f"[analyze] lista negra activa: {len(black)} detección(es) a ignorar")

    for i, photo in enumerate(photos, 1):
        if should_stop and should_stop():
            stopped = True
            log(f"[analyze] detenido por el usuario en {i}/{len(photos)}")
            break
        rel = photo.relative_to(photos_dir).as_posix()
        if progress:
            progress(i, len(photos), rel)
        log(f"[analyze] ({i}/{len(photos)}) {rel} ...", end=" ")

        if photo.suffix.lower() in VIDEO_EXTENSIONS:
            try:
                kept, sampled = _scan_video(photo, should_stop, black, min_face_px)
            except Exception as e:
                log(f"no se pudo leer: {e}")
                no_faces.append(rel)
                continue
            if not kept:
                log(f"sin rostros ({sampled} fotogramas)")
                no_faces.append(rel)
                continue
            log(f"{len(kept)} persona(s) en {sampled} fotogramas")
            for k in kept:
                face_id = f"{counter:05d}"
                counter += 1
                faces[face_id] = {
                    "photo": rel, "box": list(k["box"]), "time": round(k["t"], 1),
                }
                encodings.append(k["enc"])
                save_thumbnail(k["frame"], k["box"], face_id)
            continue

        try:
            image = _load_rgb(photo)
            dets = [d for d in detect_and_encode(image)
                    if _big_enough(d[0], min_face_px) and not is_not_face(d[1], black)]
        except Exception as e:
            # una foto ilegible o que hace fallar al detector no corta el
            # análisis: se saltea con aviso y sigue con las demás
            log(f"no se pudo procesar: {e}")
            no_faces.append(rel)
            continue

        if not dets:
            log("sin rostros")
            no_faces.append(rel)
            continue
        log(f"{len(dets)} rostro(s)")

        for box, encoding in dets:
            face_id = f"{counter:05d}"
            counter += 1
            faces[face_id] = {"photo": rel, "box": list(box)}
            encodings.append(encoding)
            save_thumbnail(image, box, face_id)

    if not faces:
        if stopped:
            return 0, 0   # detenido antes de encontrar ninguna cara
        raise ValueError("No se encontró ningún rostro en las fotos.")

    clusters = cluster_faces(faces, encodings)

    # Guardar los vectores (para reconocer y para sumar a la base al nombrar).
    enc_array = np.array(encodings, dtype=np.float32)
    np.save(ENCODINGS_NPY, enc_array)

    # Sugerencias: a cada grupo se le propone la persona conocida más parecida
    # (si hay alguna razonablemente cercana). No se etiqueta solo: el asistente
    # muestra la sugerencia ya escrita para que la confirmes o corrijas.
    known = load_known()
    n_sug = 0
    if known:
        for cluster in clusters:
            idxs = [int(fid) for fid in cluster["faces"]]
            cands = identify_ranked(known, enc_array[idxs], eps=SUGGEST_EPS, top_n=3)
            if cands:
                cluster["suggestions"] = [name for name, _ in cands]
                # distancia de cada candidato (la web la muestra como % de confianza)
                cluster["sug_dist"] = [round(float(d), 3) for _, d in cands]
                n_sug += 1

    with open(CLUSTERS_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "photos_dir": str(photos_dir),
                "faces": faces,
                "clusters": clusters,
                "no_faces": no_faces,
            },
            f, ensure_ascii=False, indent=2,
        )

    # Los nombres del análisis anterior ya no aplican: se empieza en limpio
    # (las sugerencias no son etiquetas confirmadas). Los aciertos de
    # sugerencias de la sesión anterior pasan al histórico de calidad.
    LABELS_PATH.unlink(missing_ok=True)
    archive_current()

    if n_sug:
        log(f"[analyze] {n_sug} grupo(s) con sugerencia de nombre.")
    return len(faces), len(clusters)


def main():
    cfg = load_config()
    if not cfg.get("photos_dir"):
        print("[analyze] Todavía no elegiste la carpeta de origen.")
        print("[analyze] Corré la web y elegila ahí: python -m app.flask_app")
        sys.exit(1)

    photos_dir = Path(cfg["photos_dir"])
    if not photos_dir.is_dir():
        print(f"[analyze] La carpeta de origen no existe: {photos_dir}")
        sys.exit(1)

    exclude = Path(cfg["output_dir"]) if cfg.get("output_dir") else None
    try:
        n_faces, n_clusters = run_analysis(photos_dir, exclude=exclude)
    except ValueError as e:
        print(f"[analyze] {e}")
        sys.exit(1)

    print(f"[analyze] Listo: {n_faces} rostro(s) en {n_clusters} grupo(s).")
    print("[analyze] Ahora corré la web para ponerles nombre: python -m app.flask_app")


if __name__ == "__main__":
    main()
