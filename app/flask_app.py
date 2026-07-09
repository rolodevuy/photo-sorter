"""Interfaz web local para ponerle nombre a cada grupo de rostros.

Muestra los grupos que armó app.analyze y pregunta "¿quién es esta persona?".
Con los nombres guardados, el botón "Organizar fotos" copia cada foto a
data/sorted/<nombre>/.

Solo escucha en 127.0.0.1: no es accesible desde afuera ni se conecta a nada.

Uso:
    python -m app.flask_app
Después abrí http://127.0.0.1:5000 en el navegador.
"""

import json
import threading
import webbrowser

from flask import Flask, redirect, render_template_string, request, send_from_directory, url_for

from app import CLUSTERS_PATH, FACES_DIR, LABELS_PATH, PHOTOS_DIR
from app.organizer import organize

app = Flask(__name__)

TEMPLATE = """
<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>photo-sorter</title>
<style>
  body { font-family: system-ui, sans-serif; margin: 2rem; background: #f5f5f5; }
  h1 { margin-bottom: .25rem; }
  .hint { color: #666; margin-bottom: 1.5rem; }
  .cluster { background: #fff; border-radius: 8px; padding: 1rem; margin-bottom: 1rem;
             box-shadow: 0 1px 3px rgba(0,0,0,.1); }
  .cluster.labeled { border-left: 6px solid #4caf50; }
  .faces { display: flex; flex-wrap: wrap; gap: 6px; margin: .5rem 0; }
  .faces a img { height: 90px; border-radius: 4px; display: block; }
  input[type=text] { padding: .4rem; font-size: 1rem; width: 220px; }
  button { padding: .4rem .9rem; font-size: 1rem; cursor: pointer; }
  .organize { background: #4caf50; color: #fff; border: 0; border-radius: 4px;
              padding: .6rem 1.2rem; font-size: 1.05rem; }
  .msg { background: #e8f5e9; border: 1px solid #4caf50; padding: .75rem 1rem;
         border-radius: 6px; margin-bottom: 1rem; }
  .meta { color: #888; font-size: .85rem; }
</style>
</head>
<body>
<h1>photo-sorter</h1>
<p class="hint">{{ clusters|length }} grupo(s) de rostros — {{ labeled }} con nombre.
Escribí quién es cada persona y guardá. Al final, tocá "Organizar fotos".</p>

{% if message %}<div class="msg">{{ message }}</div>{% endif %}

<form method="post" action="{{ url_for('do_organize') }}">
  <button class="organize" {% if labeled == 0 %}disabled{% endif %}>
    📁 Organizar fotos ({{ labeled }} persona(s) con nombre)
  </button>
</form>
<br>

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
</body>
</html>
"""

MAX_PREVIEW = 8


def load_clusters():
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
    data = load_clusters()
    labels = load_labels()
    faces = data["faces"]

    clusters = []
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
        clusters=clusters,
        labeled=sum(1 for c in clusters if c["name"]),
        message=request.args.get("msg", ""),
    )


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
    copied, people = organize()
    return redirect(url_for("home", msg=f"Listo: {copied} foto(s) copiadas a data/sorted/ "
                                        f"en {people} carpeta(s) de persona."))


@app.route("/face/<face_id>")
def face(face_id):
    return send_from_directory(FACES_DIR, f"{face_id}.jpg")


@app.route("/photo/<path:path>")
def photo(path):
    return send_from_directory(PHOTOS_DIR, path)


def main():
    if not CLUSTERS_PATH.is_file():
        print("[web] No existe data/clusters.json. Corré primero: python -m app.analyze")
        raise SystemExit(1)
    print("[web] Abrí http://127.0.0.1:5000 en el navegador (solo accesible desde esta máquina).")
    threading.Timer(1.5, lambda: webbrowser.open("http://127.0.0.1:5000")).start()
    app.run(host="127.0.0.1", port=5000, debug=False)


if __name__ == "__main__":
    main()
