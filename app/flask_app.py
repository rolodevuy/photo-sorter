"""Interfaz web local de photo-sorter.

Todo se maneja desde acá:
1. Elegir carpeta de ORIGEN (donde están tus fotos; no se tocan) y carpeta de
   DESTINO (donde se crean las carpetas por persona).
2. Analizar: detecta y agrupa los rostros (con barra de progreso).
3. Ponerle nombre a cada grupo, de a uno por vez ("¿quién es esta persona?").
4. Organizar: copia cada foto a <destino>/<nombre>/ como <nombre>_0000.jpg.

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
  body { font-family: system-ui, sans-serif; margin: 2rem auto; max-width: 860px;
         padding: 0 1rem; background: #f5f5f5; }
  h1 { margin-bottom: .25rem; }
  .hint { color: #666; margin-bottom: 1.5rem; }
  .panel, .card { background: #fff; border-radius: 8px; padding: 1rem;
                  margin-bottom: 1rem; box-shadow: 0 1px 3px rgba(0,0,0,.1); }
  .faces { display: flex; flex-wrap: wrap; gap: 6px; margin: .75rem 0; }
  .faces a img { height: 110px; border-radius: 4px; display: block; }
  .row { display: flex; align-items: center; gap: .5rem; margin-bottom: .6rem; flex-wrap: wrap; }
  .row label { min-width: 70px; font-weight: 600; }
  input[type=text] { padding: .45rem; font-size: 1.05rem; }
  .path { width: 420px; max-width: 85vw; }
  button { padding: .45rem .9rem; font-size: 1rem; cursor: pointer; }
  .primary { background: #1976d2; color: #fff; border: 0; border-radius: 4px; }
  .secondary { background: #eee; border: 1px solid #ccc; border-radius: 4px; }
  .organize { background: #4caf50; color: #fff; border: 0; border-radius: 4px;
              padding: .6rem 1.2rem; font-size: 1.05rem; }
  button:disabled { opacity: .5; cursor: default; }
  .msg { background: #e8f5e9; border: 1px solid #4caf50; padding: .75rem 1rem;
         border-radius: 6px; margin-bottom: 1rem; }
  .err { background: #ffebee; border: 1px solid #e53935; padding: .75rem 1rem;
         border-radius: 6px; margin-bottom: 1rem; }
  .meta { color: #888; font-size: .85rem; }
  progress { width: 100%; height: 18px; }
  #wizard-card { transition: opacity .25s, transform .25s; }
  #wizard-card.fade { opacity: 0; transform: translateY(-10px); }
  .counter { font-size: 1.1rem; font-weight: 600; margin-bottom: .5rem; }
  .done-big { font-size: 1.3rem; margin: .5rem 0; }
  #list-view .card.labeled { border-left: 6px solid #4caf50; }
  .saved-tick { color: #2e7d32; font-weight: 600; margin-left: .5rem;
                opacity: 0; transition: opacity .3s; }
  .saved-tick.show { opacity: 1; }
</style>
</head>
<body>
<h1>photo-sorter</h1>
<p class="hint">Al organizar, tus fotos se <b>mueven</b> al destino como <code>nombre_0000.jpg</code>, separadas por persona (desaparecen del origen). Las fotos sin nombre quedan donde estaban. Todo corre en esta máquina.</p>

{% if message %}<div class="msg">{{ message }}</div>{% endif %}
{% if state.status == 'error' %}<div class="err">Error del análisis: {{ state.error }}</div>{% endif %}

<div class="panel">
  <form method="post" action="{{ url_for('settings') }}">
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

{% if groups %}
<div class="panel">
  <div class="counter">Se encontraron <b>{{ groups|length }}</b> grupo(s) de rostros —
    <span id="labeled-count">{{ labeled }}</span> con nombre,
    <span id="pending-count"></span> por revisar.</div>
  <form method="post" action="{{ url_for('do_organize') }}">
    <button class="organize" id="organize-btn" {% if not cfg.get('output_dir') %}disabled{% endif %}
            onclick="return confirm('Se moverán las fotos con nombre al destino (desaparecen del origen). ¿Continuar?')">
      📁 Organizar: mover fotos al destino
    </button>
    <button type="button" class="secondary" onclick="toggleList()">Ver/editar todos los grupos</button>
    {% if not cfg.get('output_dir') %}<span class="meta">Elegí antes la carpeta de destino.</span>{% endif %}
  </form>
</div>

<div class="card" id="wizard-card"></div>

<div id="list-view" style="display:none"></div>

<script>
const GROUPS = {{ groups|tojson }};
const FACE_URL = "{{ url_for('face', face_id='FID') }}";
const PHOTO_URL = "{{ url_for('photo', path='RELPATH') }}";
const MAX_PREVIEW = 12;

let queue = GROUPS.filter(g => !g.name).map(g => g.id);   // grupos sin nombre, en orden
let pos = 0;

const byId = {};
GROUPS.forEach(g => byId[g.id] = g);

function faceImg(f) {
  return '<a href="' + PHOTO_URL.replace('RELPATH', encodeURIComponent(f.photo)) +
         '" target="_blank" title="' + f.photo + '"><img src="' +
         FACE_URL.replace('FID', f.id) + '" alt="rostro"></a>';
}

function updateCounts() {
  const labeled = GROUPS.filter(g => g.name).length;
  document.getElementById('labeled-count').textContent = labeled;
  document.getElementById('pending-count').textContent = (GROUPS.length - labeled);
}

function renderWizard() {
  const card = document.getElementById('wizard-card');
  if (pos >= queue.length) {
    const labeled = GROUPS.filter(g => g.name).length;
    card.innerHTML = '<div class="done-big">✅ No quedan grupos por revisar.</div>' +
      '<p>' + labeled + ' grupo(s) con nombre. Tocá <b>"📁 Organizar"</b> arriba para copiar las fotos al destino, ' +
      'o "Ver/editar todos los grupos" para corregir algo.</p>';
    return;
  }
  const g = byId[queue[pos]];
  const extra = g.faces.length > MAX_PREVIEW ?
    '<span class="meta">… y ' + (g.faces.length - MAX_PREVIEW) + ' más</span>' : '';
  card.innerHTML =
    '<div class="counter">Grupo ' + (pos + 1) + ' de ' + queue.length + ' por revisar</div>' +
    '<div class="meta">' + g.faces.length + ' rostro(s) en ' + g.photos + ' foto(s) — clic en una cara abre la foto completa</div>' +
    '<div class="faces">' + g.faces.slice(0, MAX_PREVIEW).map(faceImg).join('') + extra + '</div>' +
    '<div class="row">' +
      '<label>¿Quién es?</label>' +
      '<input type="text" id="wizard-name" placeholder="ej: mamá, Juan..." autofocus>' +
      '<button class="primary" onclick="wizardSave()">Guardar</button>' +
      '<button class="secondary" onclick="wizardNext()">Saltear</button>' +
    '</div>';
  const input = document.getElementById('wizard-name');
  input.focus();
  input.addEventListener('keydown', e => { if (e.key === 'Enter') wizardSave(); });
}

function saveLabel(clusterId, name) {
  return fetch("{{ url_for('label') }}", {
    method: 'POST',
    headers: {'Content-Type': 'application/x-www-form-urlencoded'},
    body: 'cluster_id=' + encodeURIComponent(clusterId) + '&name=' + encodeURIComponent(name)
  });
}

function wizardSave() {
  const name = document.getElementById('wizard-name').value.trim();
  if (!name) { wizardNext(); return; }
  const g = byId[queue[pos]];
  saveLabel(g.id, name).then(() => {
    g.name = name;
    updateCounts();
    wizardNext();
  });
}

function wizardNext() {
  const card = document.getElementById('wizard-card');
  card.classList.add('fade');
  setTimeout(() => { pos++; renderWizard(); card.classList.remove('fade'); }, 250);
}

function toggleList() {
  const lv = document.getElementById('list-view');
  if (lv.style.display === 'none') { renderList(); lv.style.display = 'block'; }
  else lv.style.display = 'none';
}

function renderList() {
  const lv = document.getElementById('list-view');
  lv.innerHTML = GROUPS.map(g =>
    '<div class="card' + (g.name ? ' labeled' : '') + '" id="lg-' + g.id + '">' +
      '<div class="meta">Grupo ' + g.id + ' — ' + g.faces.length + ' rostro(s) en ' + g.photos + ' foto(s)</div>' +
      '<div class="faces">' + g.faces.slice(0, 8).map(faceImg).join('') + '</div>' +
      '<div class="row"><label>¿Quién es?</label>' +
        '<input type="text" id="ln-' + g.id + '" value="' + (g.name || '').replace(/"/g, '&quot;') + '">' +
        '<button class="primary" onclick="listSave(' + g.id + ')">Guardar</button>' +
        '<span class="saved-tick" id="lt-' + g.id + '">✓ guardado</span>' +
      '</div>' +
    '</div>'
  ).join('');
}

function listSave(id) {
  const name = document.getElementById('ln-' + id).value.trim();
  saveLabel(id, name).then(() => {
    byId[id].name = name;
    updateCounts();
    const card = document.getElementById('lg-' + id);
    card.classList.toggle('labeled', !!name);
    const tick = document.getElementById('lt-' + id);
    tick.classList.add('show');
    setTimeout(() => tick.classList.remove('show'), 1500);
  });
}

updateCounts();
renderWizard();
</script>
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

    groups = []
    if data:
        faces = data["faces"]
        for c in data["clusters"]:
            members = c["faces"]
            groups.append({
                "id": c["id"],
                "faces": [{"id": fid, "photo": faces[fid]["photo"]} for fid in members],
                "photos": len({faces[fid]["photo"] for fid in members}),
                "name": labels.get(str(c["id"]), ""),
            })

    return render_template_string(
        TEMPLATE,
        cfg=cfg,
        state=STATE,
        groups=groups,
        labeled=sum(1 for g in groups if g["name"]),
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
    return {"ok": True, "name": name}


@app.route("/organize", methods=["POST"])
def do_organize():
    try:
        moved, people, output_dir, errors = organize()
    except ValueError as e:
        return redirect(url_for("home", msg=str(e)))
    msg = (f"Listo: {moved} foto(s) movidas a {output_dir} "
           f"en {people} carpeta(s) de persona.")
    if errors:
        msg += f" {len(errors)} no se pudieron mover y quedaron en el origen."
    return redirect(url_for("home", msg=msg))


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
