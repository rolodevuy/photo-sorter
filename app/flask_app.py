"""Interfaz web local de photo-sorter.

Todo se maneja desde acá:
1. Elegir carpeta de ORIGEN (donde están tus fotos; no se tocan) y carpeta de
   DESTINO (donde se crean las carpetas por persona).
2. Analizar: detecta y agrupa los rostros (con barra de progreso).
3. Ponerle nombre a cada grupo ("¿quién es esta persona?").
4. Organizar: copia cada foto a <destino>/<nombre>/.

Solo escucha en 127.0.0.1: no es accesible desde afuera ni se conecta a nada.

Uso:
    python -m app.flask_app
"""

import json
import threading
import webbrowser
from pathlib import Path

from flask import Flask, redirect, render_template_string, request, send_from_directory, url_for

from app import CLUSTERS_PATH, FACES_DIR, LABELS_PATH
from app.analyze import run_analysis
from app.config import load_config, save_config
from app.organizer import organize

app = Flask(__name__)

# Estado del análisis en curso (corre en un hilo aparte)
STATE = {"status": "idle", "current": 0, "total": 0, "photo": "", "error": ""}

TEMPLATE = """
<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>photo-sorter</title>
{% if state.status == 'running' %}<meta http-equiv="refresh" content="2">{% endif %}
<style>
  body { font-family: system-ui, sans-serif; margin: 2rem; background: #f5f5f5; }
  h1 { margin-bottom: .25rem; }
  .hint { color: #666; margin-bottom: 1.5rem; }
  .panel, .cluster { background: #fff; border-radius: 8px; padding: 1rem;
                     margin-bottom: 1rem; box-shadow: 0 1px 3px rgba(0,0,0,.1); }
  .cluster.labeled { border-left: 6px solid #4caf50; }
  .faces { display: flex; flex-wrap: wrap; gap: 6px; margin: .5rem 0; }
  .faces a img { height: 90px; border-radius: 4px; display: block; }
  .row { display: flex; align-items: center; gap: .5rem; margin-bottom: .6rem; flex-wrap: wrap; }
  .row label { min-width: 70px; font-weight: 600; }
  input[type=text] { padding: .4rem; font-size: 1rem; }
  .path { width: 420px; max-width: 90vw; }
  button { padding: .4rem .9rem; font-size: 1rem; cursor: pointer; }
  .primary { background: #1976d2; color: #fff; border: 0; border-radius: 4px; }
  .organize { background: #4caf50; color: #fff; border: 0; border-radius: 4px;
              padding: .6rem 1.2rem; font-size: 1.05rem; }
  button:disabled { opacity: .5; cursor: default; }
  .msg { background: #e8f5e9; border: 1px solid #4caf50; padding: .75rem 1rem;
         border-radius: 6px; margin-bottom: 1rem; }
  .err { background: #ffebee; border: 1px solid #e53935; padding: .75rem 1rem;
         border-radius: 6px; margin-bottom: 1rem; }
  .meta { color: #888; font-size: .85rem; }
  progress { width: 100%; height: 18px; }
</style>
</head>
<body>
<h1>photo-sorter</h1>
<p class="hint">Tus fotos no se mueven ni se borran: solo se <b>copian</b> al destino, separadas por persona. Todo corre en esta máquina.</p>

{% if message %}<div class="msg">{{ message }}</div>{% endif %}
{% if state.status == 'error' %}<div class="err">Error del análisis: {{ state.error }}</div>{% endif %}

<div class="panel">
  <form method="post" action="{{ url_for('settings') }}" id="cfg">
    <div class="row">
      <label>Origen:</label>
      <input class="path" type="text" name="photos_dir" value="{{ cfg.get('photos_dir','') }}"
             placeholder="Carpeta donde están tus fotos">
      <button type="button" onclick="browse('photos_dir')">📂 Elegir carpeta…</button>
    </div>
    <div class="row">
      <label>Destino:</label>
      <input class="path" type="text" name="output_dir" value="{{ cfg.get('output_dir','') }}"
             placeholder="Carpeta donde se crearán las carpetas por persona">
      <button type="button" onclick="browse('output_dir')">📂 Elegir carpeta…</button>
    </div>
    <div class="row">
      <button class="primary">Guardar carpetas</button>
    </div>
  </form>
  <form method="post" action="{{ url_for('do_analyze') }}">
    <button class="primary" {% if not cfg.get('photos_dir') or state.status == 'running' %}disabled{% endif %}>
      🔍 Analizar fotos del origen
    </button>
    <span class="meta">Detecta los rostros y agrupa las caras iguales. Con muchas fotos tarda.</span>
  </form>
  {% if state.status == 'running' %}
    <p>Analizando {{ state.current }}/{{ state.total }}: {{ state.photo }}</p>
    <progress value="{{ state.current }}" max="{{ state.total or 1 }}"></progress>
    <p class="meta">Esta página se actualiza sola cada 2 segundos.</p>
  {% endif %}
</div>

{% if clusters %}
<div class="panel">
  <p><b>{{ clusters|length }} grupo(s) de rostros</b> — {{ labeled }} con nombre.
  Escribí quién es cada persona y guardá. Los grupos sin nombre se ignoran al organizar.</p>
  <form method="post" action="{{ url_for('do_organize') }}">
    <button class="organize" {% if labeled == 0 or not cfg.get('output_dir') %}disabled{% endif %}>
      📁 Organizar: copiar fotos al destino ({{ labeled }} persona(s) con nombre)
    </button>
    {% if not cfg.get('output_dir') %}<span class="meta">Elegí antes la carpeta de destino.</span>{% endif %}
  </form>
</div>

{% for c in clusters %}
<div class="cluster {% if c.name %}labeled{% endif %}">
  <div class="meta">Grupo {{ c.id }} — {{ c.faces|length }} rostro(s) en {{ c.photos }} foto(s)</div>
  <div class="faces">
    {% for f in c.preview %}
    <a href="{{ url_for('photo', path=f.photo) }}" target="_blank" title="{{ f.photo }}">
      <img src="{{ url_for('face', face_id=f.id) }}" alt="rostro">
    </a>
    {% endfor %}
    {% if c.faces|length > c.preview|length %}
      <span class="meta">… y {{ c.faces|length - c.preview|length }} más</span>
    {% endif %}
  </div>
  <form method="post" action="{{ url_for('label') }}">
    <input type="hidden" name="cluster_id" value="{{ c.id }}">
    <label>¿Quién es esta persona?
      <input type="text" name="name" value="{{ c.name }}" placeholder="ej: mamá, Juan...">
    </label>
    <button>Guardar</button>
  </form>
</div>
{% endfor %}
{% endif %}

<script>
function browse(field) {
  fetch('{{ url_for('browse') }}')
    .then(r => r.json())
    .then(d => { if (d.path) document.getElementsByName(field)[0].value = d.path; });
}
</script>
</body>
</html>
"""

MAX_PREVIEW = 8


def load_clusters():
    if not CLUSTERS_PATH.is_file():
        return None
    with open(CLUSTERS_PATH, encoding="utf-8") as f:
        return json.load(f)


def load_labels():
    if LABELS_PATH.is_file():
        with open(LABELS_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_labels(labels):
    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)


@app.route("/")
def home():
    cfg = load_config()
    data = load_clusters() if STATE["status"] != "running" else None
    labels = load_labels()

    clusters = []
    if data:
        faces = data["faces"]
        for c in data["clusters"]:
            members = c["faces"]
            clusters.append({
                "id": c["id"],
                "faces": members,
                "photos": len({faces[fid]["photo"] for fid in members}),
                "preview": [{"id": fid, "photo": faces[fid]["photo"]} for fid in members[:MAX_PREVIEW]],
                "name": labels.get(str(c["id"]), ""),
            })

    return render_template_string(
        TEMPLATE,
        cfg=cfg,
        state=STATE,
        clusters=clusters,
        labeled=sum(1 for c in clusters if c["name"]),
        message=request.args.get("msg", ""),
    )


@app.route("/settings", methods=["POST"])
def settings():
    cfg = load_config()
    cfg["photos_dir"] = request.form.get("photos_dir", "").strip()
    cfg["output_dir"] = request.form.get("output_dir", "").strip()
    save_config(cfg)
    return redirect(url_for("home", msg="Carpetas guardadas."))


@app.route("/browse")
def browse():
    """Abre el selector de carpetas nativo de Windows (en esta máquina)."""
    from tkinter import Tk, filedialog
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askdirectory(parent=root)
    root.destroy()
    return {"path": str(Path(path)) if path else ""}


def _analysis_worker(photos_dir, exclude):
    def progress(current, total, photo):
        STATE.update(current=current, total=total, photo=photo)
    try:
        run_analysis(photos_dir, exclude=exclude, progress=progress)
        STATE["status"] = "done"
    except Exception as e:
        STATE.update(status="error", error=str(e))


@app.route("/analyze", methods=["POST"])
def do_analyze():
    if STATE["status"] == "running":
        return redirect(url_for("home"))

    cfg = load_config()
    photos_dir = Path(cfg.get("photos_dir", ""))
    if not cfg.get("photos_dir") or not photos_dir.is_dir():
        return redirect(url_for("home", msg="La carpeta de origen no existe. Elegila de nuevo."))

    exclude = Path(cfg["output_dir"]) if cfg.get("output_dir") else None
    STATE.update(status="running", current=0, total=0, photo="", error="")
    threading.Thread(target=_analysis_worker, args=(photos_dir, exclude), daemon=True).start()
    return redirect(url_for("home"))


@app.route("/label", methods=["POST"])
def label():
    labels = load_labels()
    cluster_id = request.form["cluster_id"]
    name = request.form["name"].strip()
    if name:
        labels[cluster_id] = name
    else:
        labels.pop(cluster_id, None)
    save_labels(labels)
    return redirect(url_for("home"))


@app.route("/organize", methods=["POST"])
def do_organize():
    try:
        copied, people, output_dir = organize()
    except ValueError as e:
        return redirect(url_for("home", msg=str(e)))
    return redirect(url_for("home", msg=f"Listo: {copied} foto(s) copiadas a {output_dir} "
                                        f"en {people} carpeta(s) de persona."))


@app.route("/face/<face_id>")
def face(face_id):
    return send_from_directory(FACES_DIR, f"{face_id}.jpg")


@app.route("/photo/<path:path>")
def photo(path):
    data = load_clusters()
    return send_from_directory(Path(data["photos_dir"]), path)


def main():
    print("[web] Abrí http://127.0.0.1:5000 en el navegador (solo accesible desde esta máquina).")
    threading.Timer(1.5, lambda: webbrowser.open("http://127.0.0.1:5000")).start()
    app.run(host="127.0.0.1", port=5000, debug=False)


if __name__ == "__main__":
    main()
