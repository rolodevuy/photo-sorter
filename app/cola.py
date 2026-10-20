"""Cola de trabajos pesados: hacé clic en varios y se hacen de a uno, en orden.

Un trabajo es un par (persona, tarea): "las narraciones de Rosa", "los pies de
Ana". Se encolan y un solo hilo los va sacando. La gracia es doble:

1) UNA TAREA POR VEZ SOBRE TODAS LAS FOTOS, no todas las tareas foto por foto.
   Ollama tiene OLLAMA_MAX_LOADED_MODELS=1: arrancar otro modelo descarga el
   que estaba. Yendo foto por foto (narrar, pie, comentar, siguiente foto...)
   con un modelo distinto por texto, se cambia de modelo dos veces POR FOTO —
   con 1500 fotos son 3000 cargas de varios GB desde disco. Por fases se carga
   cada modelo UNA vez. Es la misma idea que no apagar el horno entre bandeja
   y bandeja.

2) SERIAL A PROPÓSITO. Correr dos trabajos en paralelo no acelera nada (el
   cuello es el mismo CPU y el mismo Ollama) y encima haría justo el cambio de
   modelo que queremos evitar.

La cola VIVE EN DISCO (data/cola.json). Es para dejarla toda la noche: si se
corta la luz o se reinicia el server a las 4 AM, al arrancar retoma donde
estaba. Lo ya hecho tampoco se pierde nunca, porque cada foto se guarda apenas
termina (y las tareas saltean lo que ya está hecho).
"""

import json
import threading
import time

from app import DATA_DIR

COLA_PATH = DATA_DIR / "cola.json"

# Las tareas, EN EL ORDEN EN QUE TIENEN QUE CORRER. No es capricho: el pie lee
# la narración y las etiquetas, y los comentarios leen el pie para contestarle.
# Encolar los pies antes que las narraciones "funciona", pero cada pie se
# generaría su narración al vuelo y volvería el cambio de modelo por foto.
# Por eso la cola se reordena sola (ver `_orden`).
TAREAS = [
    ("narracion", "🎬 Narraciones", "Mira cada foto y la describe. Es el paso que alimenta a los otros tres."),
    ("etiquetas", "🏷 Etiquetas", "Mira cada foto y anota lo que ve."),
    ("pie", "📝 Pies", "Escribe el pie de cada foto. Usa la narración y las etiquetas."),
    ("comentarios", "💬 Comentarios", "Escribe los comentarios. Usan el pie para contestarle."),
]
_ORDEN = {clave: i for i, (clave, _, _) in enumerate(TAREAS)}
CLAVES = set(_ORDEN)

_LOCK = threading.RLock()


def _cargar():
    if not COLA_PATH.is_file():
        return {"trabajos": [], "proximo_id": 1}
    try:
        with open(COLA_PATH, encoding="utf-8") as f:
            d = json.load(f)
        d.setdefault("trabajos", [])
        d.setdefault("proximo_id", 1)
        return d
    except Exception:
        return {"trabajos": [], "proximo_id": 1}


def _guardar(d):
    COLA_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = COLA_PATH.with_suffix(".tmp")
    # escribir a un temporal y renombrar: si se corta la luz justo acá, el
    # cola.json de antes queda entero en vez de a medio escribir
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)
    tmp.replace(COLA_PATH)


def _orden(trabajos):
    """Pendientes ordenados por FASE; lo que ya está corriendo no se mueve.

    Estable dentro de la misma fase: entre dos personas con la misma tarea,
    manda el orden en que las encolaste."""
    corriendo = [t for t in trabajos if t["estado"] == "corriendo"]
    resto = [t for t in trabajos if t["estado"] != "corriendo"]
    resto.sort(key=lambda t: (t["estado"] != "pendiente",      # pendientes primero
                              _ORDEN.get(t["tarea"], 99), t["id"]))
    return corriendo + resto


def agregar(persona, tarea):
    """Encola (persona, tarea). Si ya está pendiente o corriendo, no la duplica.

    Devuelve (trabajo, reordenado): `reordenado` avisa si la cola se acomodó
    sola, para poder decírselo al usuario en vez de moverle las cosas calladito."""
    if tarea not in CLAVES:
        raise ValueError(f"Tarea desconocida: {tarea}")
    with _LOCK:
        d = _cargar()
        for t in d["trabajos"]:
            if (t["persona"] == persona and t["tarea"] == tarea
                    and t["estado"] in ("pendiente", "corriendo")):
                return t, False
        trabajo = {"id": d["proximo_id"], "persona": persona, "tarea": tarea,
                   "estado": "pendiente", "total": 0, "hechas": 0,
                   "salteadas": 0, "fallaron": 0, "error": "",
                   "creado": time.time(), "empezo": 0.0, "termino": 0.0,
                   "segundos": []}
        d["proximo_id"] += 1
        antes = [t["id"] for t in d["trabajos"] if t["estado"] == "pendiente"]
        d["trabajos"].append(trabajo)
        d["trabajos"] = _orden(d["trabajos"])
        ahora = [t["id"] for t in d["trabajos"] if t["estado"] == "pendiente"]
        _guardar(d)
        # se reordenó si el nuevo no quedó último entre los pendientes
        reordenado = bool(antes) and ahora[-1] != trabajo["id"]
        return trabajo, reordenado


def listar():
    with _LOCK:
        return list(_cargar()["trabajos"])


def siguiente():
    """El primer pendiente (ya ordenado por fase), o None."""
    with _LOCK:
        for t in _cargar()["trabajos"]:
            if t["estado"] == "pendiente":
                return t
        return None


def actualizar(id_, **campos):
    """Cambia campos de un trabajo y guarda. Devuelve el trabajo, o None."""
    with _LOCK:
        d = _cargar()
        for t in d["trabajos"]:
            if t["id"] == id_:
                t.update(campos)
                _guardar(d)
                return t
        return None


def sacar(id_):
    """Saca un trabajo de la cola. Si está corriendo NO lo mata: lo marca para
    que el worker corte solo (matar un hilo a mano deja archivos a medio
    escribir). Devuelve True si lo encontró."""
    with _LOCK:
        d = _cargar()
        for i, t in enumerate(d["trabajos"]):
            if t["id"] != id_:
                continue
            if t["estado"] == "corriendo":
                t["estado"] = "cancelando"
            else:
                d["trabajos"].pop(i)
            _guardar(d)
            return True
        return False


def limpiar_terminados():
    """Saca de la lista los que ya terminaron (listos, cortados o con error)."""
    with _LOCK:
        d = _cargar()
        antes = len(d["trabajos"])
        d["trabajos"] = [t for t in d["trabajos"]
                         if t["estado"] in ("pendiente", "corriendo", "cancelando")]
        _guardar(d)
        return antes - len(d["trabajos"])


def rescatar_huerfanos():
    """Al arrancar el server: lo que quedó marcado "corriendo" o "cancelando" de
    la vez pasada no está corriendo (el hilo murió con el proceso). Vuelve a
    pendiente para que la cola lo retome sola.

    Esta es la función que hace que dejarlo toda la noche tenga sentido: sin
    esto, un reinicio a las 4 AM dejaba un trabajo congelado en "corriendo" y la
    cola no avanzaba más."""
    with _LOCK:
        d = _cargar()
        n = 0
        for t in d["trabajos"]:
            if t["estado"] in ("corriendo", "cancelando"):
                t.update(estado="pendiente", empezo=0.0)
                n += 1
        if n:
            _guardar(d)
        return n


def resumen():
    """Lo que necesita la campana: qué corre, cuántos esperan y cuánto falta."""
    trabajos = listar()
    corriendo = next((t for t in trabajos if t["estado"] in ("corriendo", "cancelando")), None)
    pendientes = [t for t in trabajos if t["estado"] == "pendiente"]
    return {"corriendo": corriendo, "pendientes": pendientes,
            "terminados": [t for t in trabajos
                           if t["estado"] not in ("pendiente", "corriendo", "cancelando")],
            "restantes": len(pendientes) + (1 if corriendo else 0)}


def eta(trabajo):
    """Segundos que faltan para ese trabajo, con los tiempos REALES que lleva.

    Nada de estimar con un número inventado: se promedian los segundos que
    tardó cada foto de ESTE trabajo. Devuelve None hasta tener con qué (las
    primeras fotos, sin muestras, no se pueden estimar)."""
    segs = trabajo.get("segundos") or []
    total, hechas = trabajo.get("total") or 0, trabajo.get("hechas") or 0
    if not segs or not total:
        return None
    prom = sum(segs[-30:]) / len(segs[-30:])       # las últimas, no todas
    return max(0, int(prom * (total - hechas)))
