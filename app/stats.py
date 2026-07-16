"""Calidad del reconocimiento: mide cuántas sugerencias de nombre acertaron.

Cada vez que se nombra un grupo que tenía sugerencia, se anota si la primera
sugerencia era la correcta (acierto) o hubo que corregirla (error). Con eso
la web muestra el % de aciertos del análisis actual y el histórico: si el
porcentaje sube a medida que nombrás gente, la base de personas conocidas
está mejorando de verdad.

data/stats.json:
    {"current": {"<cluster_id>": true/false},
     "history": [{"date": "...", "hits": n, "misses": n}, ...]}
"""

import json
from datetime import datetime

from app import STATS_PATH


def _load():
    if not STATS_PATH.is_file():
        return {"current": {}, "history": []}
    try:
        with open(STATS_PATH, encoding="utf-8") as f:
            data = json.load(f)
        data.setdefault("current", {})
        data.setdefault("history", [])
        return data
    except Exception:
        return {"current": {}, "history": []}


def _save(data):
    STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(STATS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def record_verdict(cluster_id, hit):
    """Anota el resultado de nombrar un grupo que tenía sugerencia.

    hit=True (la primera sugerencia era la correcta), False (hubo que
    corregirla) o None (se quitó el nombre: se borra el veredicto). Nombrar
    de nuevo el mismo grupo pisa el veredicto anterior.
    """
    data = _load()
    cid = str(cluster_id)
    if hit is None:
        data["current"].pop(cid, None)
    else:
        data["current"][cid] = bool(hit)
    _save(data)


def archive_current():
    """Cierra la sesión actual: sus aciertos pasan al histórico. No hace nada
    si en la sesión no se nombró ningún grupo con sugerencia."""
    data = _load()
    verdicts = [bool(v) for v in data["current"].values()]
    if not verdicts:
        return
    data["history"].append({
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "hits": sum(verdicts),
        "misses": len(verdicts) - sum(verdicts),
    })
    data["current"] = {}
    _save(data)


def summary():
    """Resumen para la web: aciertos de la sesión actual y del histórico
    (solo sesiones anteriores, para poder comparar si estás mejorando)."""
    data = _load()
    cur = [bool(v) for v in data["current"].values()]
    hist_hits = sum(s.get("hits", 0) for s in data["history"])
    hist_misses = sum(s.get("misses", 0) for s in data["history"])
    return {
        "cur_hits": sum(cur),
        "cur_total": len(cur),
        "hist_hits": hist_hits,
        "hist_total": hist_hits + hist_misses,
    }
