"""Búsqueda por PARECIDO VISUAL: "mostrame otras fotos como esta".

Un CLIP chico (la torre de imagen nomás) convierte cada foto en un vector de
512 números que resume qué se ve. Fotos parecidas dan vectores parecidos, así
que "buscar parecidas" es ordenar por distancia coseno — instantáneo una vez que
el índice está armado.

Distinto de Duplicados, que compara los PÍXELES y solo encuentra la misma foto
otra vez (o recortada/recomprimida). Acá el parecido es de CONTENIDO: la misma
pose en otro lugar, el mismo tipo de plano, la misma ropa. De yapa, un parecido
altísimo (>= CASI_IGUAL) sí suele ser el mismo archivo repetido, así que también
sirve como detector de duplicados aproximados.

Todo local: los pesos se bajan de HuggingFace UNA vez (a la caché de HF_HOME,
que apunta a E:) y después corre sin conexión. Las fotos no salen de la máquina.

OJO, lo caro es armar el índice: hay que pasar CADA foto por el modelo, en CPU.
Es de a ratos y se puede cortar y retomar (`indexar` saltea lo ya hecho), pero
con una biblioteca grande es cuestión de horas. Por eso NADA de esto arranca
solo: lo dispara el usuario desde Ajustes, avisado del costo.
"""

import json
import os
import threading
from pathlib import Path

import numpy as np
from PIL import Image

from app import DATA_DIR

INDICE_PATH = DATA_DIR / "similar.npz"

# ViT-B/32 en vez de B/16: 4 veces menos parches por imagen, o sea ~4x más
# rápido en CPU, y para "¿se parecen?" la diferencia de calidad no se nota.
MODELO = "vit_base_patch32_clip_224.openai"

# De acá para arriba ya no es "parecida" sino casi seguro la MISMA foto
# (recomprimida, recortada, otro tamaño).
CASI_IGUAL = 0.95

_MODEL = None
_TRANSFORM = None
_LOCK = threading.Lock()

# Igual que en sexpos.py: si los pesos ya están en la caché, modo offline, así
# timm no consulta el Hub cada vez que carga el modelo en frío. Va acá arriba
# porque huggingface_hub lee la variable al importarse, y timm se importa
# perezosamente más abajo.
_CACHE_HF = Path(os.environ.get("HF_HOME") or (Path.home() / ".cache" / "huggingface"))
if (_CACHE_HF / "hub" / "models--timm--vit_base_patch32_clip_224.openai").is_dir():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")


def disponible():
    """True si timm está instalado (los pesos se bajan solos la primera vez).

    find_spec y NO `import timm`: importar timm arrastra torch entero y tarda
    ~9 segundos. Como Ajustes pregunta esto en cada carga, la página se
    arrastraba por eso. find_spec solo mira si el paquete existe en disco, sin
    ejecutar nada."""
    import importlib.util
    return importlib.util.find_spec("timm") is not None


def descargado():
    """True si los pesos ya están en la caché: si no, el primer uso baja ~350 MB.
    Sirve para avisarlo ANTES en vez de dejar la web colgada sin explicación."""
    return (_CACHE_HF / "hub" / f"models--timm--{MODELO}").is_dir()


def _cargar():
    global _MODEL, _TRANSFORM
    if _MODEL is not None:
        return
    import timm
    from timm.data import create_transform, resolve_data_config
    m = timm.create_model(f"hf_hub:timm/{MODELO}", pretrained=True, num_classes=0)
    m.eval()
    _TRANSFORM = create_transform(**resolve_data_config({}, model=m))
    _MODEL = m


def vector(image_path):
    """El vector de una foto, ya normalizado (norma 1), para que comparar sea un
    producto punto. Lanza si la imagen o el modelo fallan."""
    import torch
    with _LOCK:
        _cargar()
        img = Image.open(image_path).convert("RGB")
        x = _TRANSFORM(img).unsqueeze(0)
        with torch.no_grad():
            v = _MODEL(x)[0].numpy().astype("float32")
    n = float(np.linalg.norm(v))
    return v / n if n else v


# ── El índice: claves "<persona>/<rel>" + matriz de vectores ────────────────

def _cargar_indice():
    """(claves, matriz). Vacío si no hay índice o si lo armó otro modelo (un
    índice mezclado no compara nada: mejor rehacerlo)."""
    if not INDICE_PATH.is_file():
        return [], np.zeros((0, 0), dtype="float32")
    try:
        d = np.load(INDICE_PATH, allow_pickle=False)
        if str(d.get("modelo", "")) and str(d["modelo"]) != MODELO:
            return [], np.zeros((0, 0), dtype="float32")
        # str() de verdad, no np.str_: los que salen de acá viajan a JSON y a
        # claves de diccionario, y numpy ahí molesta
        return [str(k) for k in d["claves"]], d["vecs"].astype("float32")
    except Exception:
        return [], np.zeros((0, 0), dtype="float32")


def _guardar_indice(claves, vecs):
    INDICE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(INDICE_PATH, claves=np.array(claves, dtype=object).astype(str),
             vecs=vecs.astype("float32"), modelo=np.array(MODELO))


def estado():
    """Qué hay armado hoy: {"n": cuántas fotos, "modelo": cuál, "listo": bool}."""
    claves, _ = _cargar_indice()
    return {"n": len(claves), "modelo": MODELO, "listo": bool(claves),
            "descargado": descargado()}


def borrar_indice():
    """Tira el índice (no toca ninguna foto). Para rehacerlo de cero."""
    try:
        INDICE_PATH.unlink()
        return True
    except FileNotFoundError:
        return False


def faltan(pares):
    """De [(clave, ruta)], las que todavía no están en el índice. Para saber
    cuánto falta ANTES de arrancar (y poder estimar el tiempo)."""
    ya = set(_cargar_indice()[0])
    return [(k, p) for k, p in pares if k not in ya]


def indexar(pares, progress=None, should_stop=None, cada=25):
    """Suma al índice las fotos que le falten. [(clave, ruta)] -> cuántas sumó.

    Incremental y cortable: las ya indexadas se saltean, y se guarda cada `cada`
    fotos, así cortar a la mitad no pierde lo hecho. Una foto ilegible se saltea
    sin cortar todo el lote.

    Las fotos que ya no están en disco se limpian del índice de paso — es la
    única "limpieza" automática que hace, y es sobre el índice (dato derivado,
    se puede rehacer), NUNCA sobre etiquetas ni archivos."""
    claves, vecs = _cargar_indice()
    ya = {k: i for i, k in enumerate(claves)}
    pendientes = [(k, p) for k, p in pares if k not in ya]
    nuevas_k, nuevas_v, hechas = [], [], 0

    def volcar():
        nonlocal claves, vecs, nuevas_k, nuevas_v
        if not nuevas_k:
            return
        m = np.array(nuevas_v, dtype="float32")
        vecs = m if vecs.size == 0 else np.vstack([vecs, m])
        claves = claves + nuevas_k
        _guardar_indice(claves, vecs)
        nuevas_k, nuevas_v = [], []

    for i, (k, path) in enumerate(pendientes, 1):
        if should_stop and should_stop():
            break
        if progress:
            progress(i, len(pendientes), k)
        try:
            nuevas_v.append(vector(path))
            nuevas_k.append(k)
            hechas += 1
        except Exception:
            continue          # ilegible o formato raro: se saltea
        if len(nuevas_k) >= cada:
            volcar()
    volcar()

    # las que ya no existen: fuera del índice (solo si el barrido fue completo,
    # para no borrar lo de una persona porque se indexó a otra)
    vivas = {k for k, _ in pares}
    if vivas and not (should_stop and should_stop()):
        quedan = [i for i, k in enumerate(claves) if k in vivas]
        if len(quedan) != len(claves):
            claves = [claves[i] for i in quedan]
            vecs = vecs[quedan] if len(quedan) else np.zeros((0, 0), dtype="float32")
            _guardar_indice(claves, vecs)
    return hechas


# ── Casi iguales: duplicados APROXIMADOS ────────────────────────────────────
# Duplicados (app/duplicates.py) compara píxeles: encuentra el MISMO archivo.
# Esto compara los vectores del índice: encuentra la misma foto aunque esté
# recomprimida, recortada o en otro tamaño — lo que aquel no puede ver.

IGNORAR_PATH = DATA_DIR / "parecidos_ignorar.json"


def _par_clave(a, b):
    """Clave estable de un par, sin importar el orden en que llegue."""
    return "|".join(sorted((a, b)))


def pares_ignorados():
    """Los pares que marcaste "no son iguales": no se vuelven a mostrar."""
    try:
        with open(IGNORAR_PATH, encoding="utf-8") as f:
            return set(json.load(f))
    except Exception:
        return set()


def ignorar_par(a, b):
    """Recuerda que ese par NO es un duplicado. No toca ninguna foto."""
    pares = pares_ignorados()
    pares.add(_par_clave(a, b))
    IGNORAR_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(IGNORAR_PATH, "w", encoding="utf-8") as f:
        json.dump(sorted(pares), f, ensure_ascii=False, indent=2)
    return len(pares)


def casi_iguales(umbral=None, tope=200):
    """Pares del índice con parecido >= umbral (CASI_IGUAL por defecto):
    [(claveA, claveB, parecido)] de más a menos parecido, sin los ignorados.

    Por BLOQUES a propósito: la matriz n×n entera con miles de fotos son cientos
    de MB; en tiras de 512 filas el pico de memoria queda chico y el resultado
    es el mismo. `tope` corta la lista (200 pares ya son una tarde de revisar)."""
    claves, vecs = _cargar_indice()
    if len(claves) < 2 or vecs.size == 0:
        return []
    umbral = CASI_IGUAL if umbral is None else float(umbral)
    fuera = pares_ignorados()
    out = []
    B = 512
    for i0 in range(0, len(claves), B):
        bloque = vecs[i0:i0 + B] @ vecs.T          # (B, n) de cosenos
        for bi in range(bloque.shape[0]):
            i = i0 + bi
            for j in np.where(bloque[bi] >= umbral)[0]:
                j = int(j)
                if j <= i:                          # cada par una sola vez
                    continue
                if _par_clave(claves[i], claves[j]) in fuera:
                    continue
                out.append((claves[i], claves[j], round(float(bloque[bi][j]), 4)))
    out.sort(key=lambda p: -p[2])
    return out[:tope]


def parecidas(clave, n=12, minimo=0.55):
    """Las n fotos más parecidas a esa, sin contarla a ella:
    [(clave, parecido 0..1)] de mayor a menor. Vacío si no está en el índice.

    minimo corta la cola: sin eso siempre devuelve n resultados, aunque el
    parecido sea de casualidad."""
    claves, vecs = _cargar_indice()
    if not claves or vecs.size == 0:
        return []
    try:
        i = claves.index(clave)
    except ValueError:
        return []
    sims = vecs @ vecs[i]                  # todos normalizados: producto = coseno
    orden = np.argsort(-sims)
    out = []
    for j in orden:
        if int(j) == i:
            continue
        s = float(sims[int(j)])
        if s < minimo:
            break                          # viene ordenado: de acá para abajo, peor
        out.append((claves[int(j)], round(s, 4)))
        if len(out) >= n:
            break
    return out
