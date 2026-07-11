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

import numpy as np

from app import (CLUSTERS_PATH, DUPES_PATH, DUPES_THUMBS, ENCODINGS_NPY, FACES_DIR,
                 KNOWN_PATH, KNOWN_THUMBS, LABELS_PATH)
from app.analyze import run_analysis
from app.config import load_config, save_config
from app.duplicates import DEFAULT_THRESHOLD, find_duplicates, resolve_duplicates
from app.known import (add_faces, enroll_from_folder, forget_person, known_summary,
                       load_known, save_known, safe_name)
from app.organizer import organize
from app.renamer import apply_rename, plan_rename

# Marcado de caras "dudosas" (podrían no ser esa persona). Para cada cara se
# mide su distancia al centro del grupo y se marca si se despega claramente del
# resto: distancia > mediana del grupo + margen, con un piso absoluto. Adaptativo
# porque hay personas que varían más que otras. Solo en grupos de 3+ caras.
SUSPECT_FLOOR = 0.40
SUSPECT_MARGIN = 0.20

app = Flask(__name__)

# Estado del análisis en curso (corre en un hilo aparte)
STATE = {"status": "idle", "current": 0, "total": 0, "photo": "", "error": ""}
# Estado de la búsqueda de duplicados (otro hilo)
DUPE_STATE = {"status": "idle", "current": 0, "total": 0, "photo": "", "error": ""}
# Estado de la importación de personas conocidas (otro hilo)
IMPORT_STATE = {"status": "idle", "current": 0, "total": 0, "photo": "", "error": ""}

TEMPLATE = """
<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>photo-sorter</title>
{% if state.status == 'running' or imp.status == 'running' %}<meta http-equiv="refresh" content="2">{% endif %}
<style>
  body { font-family: system-ui, sans-serif; margin: 2rem auto; max-width: 860px;
         padding: 0 1rem; background: #f5f5f5; }
  h1 { margin-bottom: .25rem; }
  .hint { color: #666; margin-bottom: 1.5rem; }
  .panel, .card { background: #fff; border-radius: 8px; padding: 1rem;
                  margin-bottom: 1rem; box-shadow: 0 1px 3px rgba(0,0,0,.1); }
  .faces { display: flex; flex-wrap: wrap; gap: 6px; margin: .75rem 0; }
  .faces a img { height: 110px; border-radius: 4px; display: block; }
  .facewrap { position: relative; display: inline-block; }
  .facewrap .fx { position: absolute; top: 3px; right: 3px; background: #e53935;
                  color: #fff; border-radius: 50%; width: 22px; height: 22px;
                  line-height: 22px; text-align: center; font-size: .8rem;
                  cursor: pointer; opacity: .8; user-select: none; }
  .facewrap .fx:hover { opacity: 1; }
  .facewrap.suspect a img { outline: 3px solid #ff9800; }
  .facewrap.suspect .warn { position: absolute; top: 3px; left: 3px; background: #ff9800;
                            color: #fff; border-radius: 4px; font-size: .7rem; padding: 0 4px; }
  .suspect-note { color: #e65100; font-weight: 600; }
  .suggest-note { color: #1565c0; margin: .3rem 0; }
  .ask { font-size: 1.15rem; margin-right: .3rem; }
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
  /* modal para depurar un grupo */
  #modal-bg { display: none; position: fixed; inset: 0; background: rgba(0,0,0,.55);
              z-index: 50; }
  #modal { background: #fff; border-radius: 8px; max-width: 900px; width: 92vw;
           margin: 3vh auto; max-height: 94vh; display: flex; flex-direction: column; }
  #modal-head { padding: 1rem; border-bottom: 1px solid #eee; }
  #modal-body { padding: 1rem; overflow-y: auto; }
  #modal-foot { padding: 1rem; border-top: 1px solid #eee; display: flex;
                gap: .5rem; align-items: center; flex-wrap: wrap; }
  .pick { display: inline-block; position: relative; cursor: pointer; }
  .pick img { height: 120px; border-radius: 4px; display: block; border: 3px solid transparent; }
  .pick.sel img { border-color: #e53935; }
  .pick .x { position: absolute; top: 4px; right: 4px; background: #e53935; color: #fff;
             border-radius: 50%; width: 22px; height: 22px; text-align: center;
             line-height: 22px; font-size: .8rem; opacity: 0; }
  .pick.sel .x { opacity: 1; }
  .pick.suspectpick img { outline: 3px solid #ff9800; }
  .pick .warn2 { position: absolute; top: 4px; left: 4px; background: #ff9800; color:#fff;
                 font-size: .7rem; padding: 0 4px; border-radius: 4px; }
  .pick .pn { font-size: .7rem; color: #999; text-align: center; max-width: 120px;
              overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .danger { background: #e53935; color: #fff; border: 0; border-radius: 4px; }
</style>
</head>
<body>
<h1>photo-sorter</h1>
<p><b>Ordenar por rostro</b> · <a href="{{ url_for('dupes_page') }}">Buscar duplicados</a> · <a href="{{ url_for('rename_page') }}">Corregir carpeta</a></p>
<p class="hint">Al organizar, tus fotos se <b>mueven</b> al destino como <code>nombre_0000.jpg</code>, separadas por persona (desaparecen del origen). Las fotos sin nombre quedan donde estaban. Todo corre en esta máquina.</p>

{% if message %}<div class="msg">{{ message }}</div>{% endif %}
{% if state.status == 'error' %}<div class="err">Error del análisis: {{ state.error }}</div>{% endif %}
{% if state.status == 'done' and groups %}<div class="msg">✅ Análisis terminado: {{ groups|length }} grupo(s) de rostros.</div>{% endif %}
{% if state.status == 'stopped' %}<div class="msg">⏹ Análisis detenido. {% if groups %}Se guardaron {{ groups|length }} grupo(s) de las fotos procesadas hasta ese punto.{% else %}No se llegó a procesar ninguna foto con rostro.{% endif %}</div>{% endif %}

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
      <button class="secondary" formaction="{{ url_for('settings') }}">💾 Guardar carpetas</button>
      <button class="primary" formaction="{{ url_for('do_analyze') }}"
              {% if state.status == 'running' %}disabled{% endif %}
              {% if groups %}onclick="return confirm('Volver a analizar rehace TODO desde cero y borra los nombres y ajustes de este análisis (las personas conocidas se conservan). ¿Seguir?')"{% endif %}>
        🔍 Analizar fotos del origen
      </button>
    </div>
    <span class="meta">"Analizar" usa la carpeta que ves arriba. Detecta los rostros (incluso de perfil o anguladas) y agrupa las caras iguales. Con muchas fotos tarda.</span>
  </form>
  {% if state.status == 'running' %}
    <p>Analizando <b>{{ state.current }}/{{ state.total }}</b>: {{ state.photo }}</p>
    <progress value="{{ state.current }}" max="{{ state.total or 1 }}"></progress>
    <form method="post" action="{{ url_for('stop_analyze') }}" style="margin-top:.5rem">
      <button class="danger" onclick="return confirm('¿Detener el análisis? Se guardan los grupos de las fotos ya procesadas.')">
        ⛔ Detener análisis
      </button>
      {% if state.cancel %}<span class="meta">deteniendo… (termina la foto actual)</span>{% endif %}
    </form>
    <p class="meta">Esta página se actualiza sola cada 2 segundos. Cerrar el navegador NO detiene el análisis.</p>
  {% endif %}
</div>

<div class="panel" {% if imp.status == 'running' %}data-refresh="1"{% endif %}>
  <div class="counter">👤 Personas conocidas: <b>{{ known_count }}</b>
    {% if known_count > 0 %}<a href="{{ url_for('known_page') }}" style="font-size:.9rem;margin-left:.5rem">ver galería →</a>{% endif %}</div>
  <p class="meta">En cada análisis, el programa sugiere el nombre de estas personas en los grupos que se parecen (vos confirmás o corregís).
     Cada persona que nombres se suma acá.</p>
  <form method="post" action="{{ url_for('known_import') }}">
    <div class="row">
      <input class="path" type="text" name="known_dir" value="{{ known_dir }}"
             placeholder="Carpeta ya ordenada (subcarpetas = nombres de personas)">
      <button type="button" onclick="browse('known_dir')">📂 Elegir carpeta…</button>
    </div>
    <div class="row">
      <button class="primary" {% if imp.status == 'running' %}disabled{% endif %}>
        📥 Importar personas de esa carpeta
      </button>
      {% if known_count > 0 %}
      <button class="secondary" formaction="{{ url_for('known_clear') }}"
              onclick="return confirm('¿Vaciar la base de personas conocidas?')">Vaciar base</button>
      {% endif %}
    </div>
    <span class="meta">Si ya tenés carpetas con caras clasificadas (una carpeta por persona), importalas y no hace falta reetiquetar.</span>
  </form>
  {% if imp.status == 'running' %}
    <p>Aprendiendo {{ imp.current }}/{{ imp.total }}: {{ imp.photo }}</p>
    <progress value="{{ imp.current }}" max="{{ imp.total or 1 }}"></progress>
    <p class="meta">Esta página se actualiza sola cada 2 segundos.</p>
  {% elif imp.status == 'error' %}
    <div class="err">Error al importar: {{ imp.error }}</div>
  {% elif imp.status == 'done' and imp.get('added') %}
    <div class="msg">Importadas {{ imp.added|length }} persona(s):
      {{ imp.added.keys()|list|join(', ') }}.</div>
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

<div id="modal-bg" onclick="if(event.target.id==='modal-bg')closeModal()">
  <div id="modal">
    <div id="modal-head">
      <div class="counter" id="modal-title"></div>
      <div class="meta">Tocá las caras que <b>no son</b> esta persona para marcarlas (borde rojo). Al quitarlas, salen de este grupo y quedan en un grupo aparte <b>"Para revisar"</b> para la segunda pasada. No se borra ninguna foto del disco.</div>
    </div>
    <div id="modal-body"><div class="faces" id="modal-faces"></div></div>
    <div id="modal-foot">
      <button class="danger" onclick="removeSelected()">🗑 Quitar seleccionadas (<span id="sel-count">0</span>)</button>
      <button class="secondary" onclick="toggleAll()">Marcar / desmarcar todas</button>
      <button class="secondary" onclick="closeModal()">Cerrar</button>
    </div>
  </div>
</div>

<script>
const GROUPS = {{ groups|tojson }};
const FACE_URL = "{{ url_for('face', face_id='FID') }}";
const PHOTO_URL = "{{ url_for('photo', path='RELPATH') }}";
const MAX_PREVIEW = 12;

let queue = GROUPS.filter(g => !g.name).map(g => g.id);   // grupos sin nombre, en orden
let pos = 0;
let wizardManual = false;   // true = mostrar el campo de texto (elegiste "Otra")
let sugIdx = 0;             // candidato actual dentro de las sugerencias del grupo

function curSuggestion(g) {
  return (g.suggestions && sugIdx < g.suggestions.length) ? g.suggestions[sugIdx] : null;
}

const byId = {};
GROUPS.forEach(g => byId[g.id] = g);

function faceImg(f, gid) {
  return '<span class="facewrap' + (f.suspect ? ' suspect' : '') + '">' +
    (f.suspect ? '<span class="warn" title="Podría no ser esta persona">?</span>' : '') +
    '<a href="' + PHOTO_URL.replace('RELPATH', encodeURIComponent(f.photo)) +
      '" target="_blank" title="' + f.photo + '"><img src="' +
      FACE_URL.replace('FID', f.id) + '" alt="rostro"></a>' +
    (gid !== undefined ?
      '<span class="fx" data-gid="' + gid + '" data-fid="' + f.id +
        '" title="Sacar esta cara del grupo (va a Para revisar)">✕</span>' : '') +
  '</span>';
}

// clic en la ✕ de una cara (delegado, así vale para wizard y lista)
document.addEventListener('click', function(ev) {
  const fx = ev.target.closest && ev.target.closest('.fx');
  if (!fx || !fx.dataset.fid) return;
  ev.preventDefault(); ev.stopPropagation();
  removeFace(parseInt(fx.dataset.gid, 10), fx.dataset.fid);
});

function discardGroup(clusterId) {
  fetch("{{ url_for('cluster_discard') }}", {
    method: 'POST',
    headers: {'Content-Type': 'application/x-www-form-urlencoded'},
    body: 'cluster_id=' + encodeURIComponent(clusterId)
  }).then(() => {
    const idx = GROUPS.findIndex(g => g.id === clusterId);
    if (idx >= 0) GROUPS.splice(idx, 1);
    delete byId[clusterId];
    queue = queue.filter(id => id !== clusterId);
    updateCounts();
    const lv = document.getElementById('list-view');
    if (lv && lv.style.display !== 'none') renderList();
    renderWizard();
  });
}

function removeFace(clusterId, faceId) {
  const g = byId[clusterId];
  if (!g) return;
  fetch("{{ url_for('cluster_remove') }}", {
    method: 'POST',
    headers: {'Content-Type': 'application/x-www-form-urlencoded'},
    body: 'cluster_id=' + encodeURIComponent(clusterId) + '&face_ids=' + encodeURIComponent(faceId)
  }).then(() => {
    g.faces = g.faces.filter(f => f.id !== faceId);
    if (g.faces.length === 0) {          // grupo vacío: sale de la cola
      queue = queue.filter(id => id !== clusterId);
    }
    updateCounts();
    const lv = document.getElementById('list-view');
    if (lv && lv.style.display !== 'none') renderList();
    renderWizard();
  });
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
    '<div class="counter">Grupo ' + (pos + 1) + ' de ' + queue.length + ' por revisar' +
      (g.review ? ' · 🔁 Para revisar (caras apartadas de otros grupos)' : '') + '</div>' +
    '<div class="meta">' + g.faces.length + ' rostro(s) en ' + g.photos + ' foto(s) — clic en una cara abre la foto completa' +
      (g.suspects ? ' · <span class="suspect-note">⚠ ' + g.suspects + ' cara(s) dudosa(s)</span>' : '') + '</div>' +
    '<div class="faces">' + g.faces.slice(0, MAX_PREVIEW).map(f => faceImg(f, g.id)).join('') + extra + '</div>' +
    '<div class="row">' +
      (g.faces.length > 10 || g.suspects ?
        '<button class="secondary" onclick="openModal(' + g.id + ')">🔍 Revisar y depurar las ' + g.faces.length + ' caras</button>' : '') +
      '<button class="secondary" onclick="discardGroup(' + g.id + ')" title="No es una cara real (tapizado, objeto, mancha)">🚫 No es una persona</button>' +
    '</div>' +
    ((curSuggestion(g) && !wizardManual) ?
      // hay candidato: preguntar ¿Es X? con Sí / No / Otra
      '<div class="row"><span class="ask">¿Es <b>' + curSuggestion(g) + '</b>?</span>' +
        '<button class="primary" onclick="wizardConfirm()">✅ Sí</button>' +
        '<button class="secondary" onclick="wizardReject()">✖ No</button>' +
        '<button class="secondary" onclick="wizardOther()">✏️ Otra persona</button>' +
      '</div>' +
      (g.suggestions && g.suggestions.length > 1 ?
        '<div class="meta">candidato ' + (sugIdx + 1) + ' de ' + g.suggestions.length + ' — "No" prueba el siguiente</div>' : '')
    :
      // sin candidatos (o elegiste "Otra"): campo para escribir el nombre
      '<div class="row">' +
        '<label>¿Quién es?</label>' +
        '<input type="text" id="wizard-name" placeholder="ej: mamá, Juan..." autofocus>' +
        '<button class="primary" onclick="wizardSave()">Guardar</button>' +
        '<button class="secondary" onclick="wizardNext()">Saltear</button>' +
      '</div>');
  const input = document.getElementById('wizard-name');
  if (input) {
    input.focus();
    input.addEventListener('keydown', e => { if (e.key === 'Enter') wizardSave(); });
  }
}

function saveLabel(clusterId, name) {
  return fetch("{{ url_for('label') }}", {
    method: 'POST',
    headers: {'Content-Type': 'application/x-www-form-urlencoded'},
    body: 'cluster_id=' + encodeURIComponent(clusterId) + '&name=' + encodeURIComponent(name)
  });
}

function wizardConfirm() {   // "Sí": confirmar el candidato actual
  const g = byId[queue[pos]];
  const sug = curSuggestion(g);
  if (!sug) return;
  saveLabel(g.id, sug).then(() => {
    g.name = sug;
    updateCounts();
    wizardNext();
  });
}

function wizardReject() {   // "No": descartar este candidato y ofrecer el siguiente
  const g = byId[queue[pos]];
  sugIdx++;
  if (curSuggestion(g)) {
    renderWizard();        // mismo grupo, próximo candidato
  } else {
    wizardNext();          // no quedan candidatos: saltear el grupo
  }
}

function wizardOther() {   // "Otra persona": mostrar el campo de texto
  wizardManual = true;
  renderWizard();
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

function wizardNext() {   // "Saltear" / avanzar tras guardar
  wizardManual = false;
  sugIdx = 0;
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
  lv.innerHTML = GROUPS.filter(g => g.faces.length > 0).map(g =>
    '<div class="card' + (g.name ? ' labeled' : '') + '" id="lg-' + g.id + '">' +
      '<div class="meta">Grupo ' + g.id + ' — ' + g.faces.length + ' rostro(s) en ' + g.photos + ' foto(s)' +
        (g.suggestions && g.suggestions.length && !g.name ? ' · <b>💡 sugerencia: ' + g.suggestions[0] + '</b>' : '') +
        (g.review ? ' · 🔁 Para revisar' : '') +
        (g.suspects ? ' · <span class="suspect-note">⚠ ' + g.suspects + ' dudosa(s)</span>' : '') + '</div>' +
      '<div class="faces">' + g.faces.slice(0, 8).map(f => faceImg(f, g.id)).join('') + '</div>' +
      '<div class="row">' +
        (g.faces.length > 10 || g.suspects ?
          '<button class="secondary" onclick="openModal(' + g.id + ')">🔍 Revisar y depurar las ' + g.faces.length + ' caras</button>' : '') +
        '<button class="secondary" onclick="discardGroup(' + g.id + ')" title="No es una cara real">🚫 No es una persona</button>' +
      '</div>' +
      '<div class="row"><label>¿Quién es?</label>' +
        '<input type="text" id="ln-' + g.id + '" value="' + (g.name || (g.suggestions && g.suggestions[0]) || '').replace(/"/g, '&quot;') + '">' +
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

// ---- modal para depurar un grupo ----
let modalGroup = null;
const selected = new Set();

function openModal(id) {
  modalGroup = byId[id];
  selected.clear();
  document.getElementById('modal-title').textContent =
    'Depurar grupo' + (modalGroup.name || (modalGroup.suggestions && modalGroup.suggestions[0]) ? ' "' + (modalGroup.name || modalGroup.suggestions[0]) + '"' : '') +
    ' — ' + modalGroup.faces.length + ' caras';
  const box = document.getElementById('modal-faces');
  // dudosas primero, para que salten a la vista
  const ordered = modalGroup.faces.slice().sort((a,b) => (b.suspect?1:0) - (a.suspect?1:0));
  box.innerHTML = ordered.map(f =>
    '<div class="pick' + (f.suspect ? ' suspectpick' : '') + '" data-fid="' + f.id + '" onclick="togglePick(this)">' +
      '<span class="x">✕</span>' +
      (f.suspect ? '<span class="warn2">⚠ dudosa</span>' : '') +
      '<img src="' + FACE_URL.replace('FID', f.id) + '" alt="cara">' +
      '<div class="pn" title="' + f.photo + '">' + f.photo.split('/').pop() + '</div>' +
    '</div>'
  ).join('');
  updateSel();
  document.getElementById('modal-bg').style.display = 'block';
}

function togglePick(el) {
  const fid = el.dataset.fid;
  if (selected.has(fid)) { selected.delete(fid); el.classList.remove('sel'); }
  else { selected.add(fid); el.classList.add('sel'); }
  updateSel();
}

function toggleAll() {
  const picks = [...document.querySelectorAll('#modal-faces .pick')];
  const allSel = picks.every(p => selected.has(p.dataset.fid));
  picks.forEach(p => {
    if (allSel) { selected.delete(p.dataset.fid); p.classList.remove('sel'); }
    else { selected.add(p.dataset.fid); p.classList.add('sel'); }
  });
  updateSel();
}

function updateSel() { document.getElementById('sel-count').textContent = selected.size; }
function closeModal() { document.getElementById('modal-bg').style.display = 'none'; }

function removeSelected() {
  if (!selected.size) { closeModal(); return; }
  if (selected.size >= modalGroup.faces.length) {
    alert('No podés quitar todas las caras del grupo. Dejá al menos una, o mejor dejá el grupo sin nombre.');
    return;
  }
  fetch("{{ url_for('cluster_remove') }}", {
    method: 'POST',
    headers: {'Content-Type': 'application/x-www-form-urlencoded'},
    body: 'cluster_id=' + encodeURIComponent(modalGroup.id) +
          '&face_ids=' + encodeURIComponent([...selected].join(','))
  }).then(() => location.reload());
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


DUPES_TEMPLATE = """
<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>photo-sorter — duplicados</title>
{% if state.status == 'running' %}<meta http-equiv="refresh" content="2">{% endif %}
<style>
  body { font-family: system-ui, sans-serif; margin: 2rem auto; max-width: 900px;
         padding: 0 1rem; background: #f5f5f5; }
  h1 { margin-bottom: .25rem; }
  .hint { color: #666; margin-bottom: 1.5rem; }
  .panel, .group { background: #fff; border-radius: 8px; padding: 1rem;
                   margin-bottom: 1rem; box-shadow: 0 1px 3px rgba(0,0,0,.1); }
  .row { display: flex; align-items: center; gap: .5rem; margin-bottom: .6rem; flex-wrap: wrap; }
  input[type=text] { padding: .45rem; font-size: 1rem; }
  .path { width: 420px; max-width: 85vw; }
  button { padding: .45rem .9rem; font-size: 1rem; cursor: pointer; }
  .primary { background: #1976d2; color: #fff; border: 0; border-radius: 4px; }
  .danger { background: #e53935; color: #fff; border: 0; border-radius: 4px;
            padding: .6rem 1.2rem; font-size: 1.05rem; }
  button:disabled { opacity: .5; cursor: default; }
  .msg { background: #e8f5e9; border: 1px solid #4caf50; padding: .75rem 1rem;
         border-radius: 6px; margin-bottom: 1rem; }
  .err { background: #ffebee; border: 1px solid #e53935; padding: .75rem 1rem;
         border-radius: 6px; margin-bottom: 1rem; }
  .meta { color: #888; font-size: .85rem; }
  progress { width: 100%; height: 18px; }
  .cards { display: flex; flex-wrap: wrap; gap: 12px; }
  .thumb { text-align: center; font-size: .8rem; }
  .thumb img { max-height: 160px; max-width: 200px; border-radius: 4px; display: block; }
  .thumb.keeper img { outline: 4px solid #4caf50; }
  .badge { display: inline-block; padding: 1px 6px; border-radius: 10px; color: #fff; font-size: .72rem; }
  .b-keep { background: #4caf50; } .b-exact { background: #8e24aa; } .b-sim { background: #fb8c00; }
</style>
</head>
<body>
<h1>photo-sorter</h1>
<p><a href="{{ url_for('home') }}">Ordenar por rostro</a> · <b>Buscar duplicados</b> · <a href="{{ url_for('rename_page') }}">Corregir carpeta</a></p>
<p class="hint">Encuentra fotos repetidas: <b>exactas</b> (mismo archivo con otro nombre) y <b>parecidas</b> (la misma imagen reescalada o recomprimida). En cada grupo se marca cuál conviene conservar (mayor resolución). Al resolver, las sobrantes se <b>mueven</b> a una carpeta <code>_duplicados</code> dentro del origen — no se borran, las revisás vos.</p>

{% if message %}<div class="msg">{{ message }}</div>{% endif %}
{% if state.status == 'error' %}<div class="err">Error: {{ state.error }}</div>{% endif %}

<div class="panel">
  <div class="row">
    <label><b>Carpeta:</b></label>
    <input class="path" type="text" value="{{ folder }}" readonly
           placeholder="Elegí la carpeta de origen en la pestaña de rostros">
    <span class="meta">(se usa la misma carpeta de origen)</span>
  </div>
  <form method="post" action="{{ url_for('dupes_scan') }}">
    <button class="primary" {% if not folder or state.status == 'running' %}disabled{% endif %}>
      🔎 Buscar duplicados
    </button>
    <span class="meta">Con muchas fotos tarda (lee cada imagen una vez).</span>
  </form>
  {% if state.status == 'running' %}
    <p>Revisando {{ state.current }}/{{ state.total }}: {{ state.photo }}</p>
    <progress value="{{ state.current }}" max="{{ state.total or 1 }}"></progress>
    <p class="meta">Esta página se actualiza sola cada 2 segundos.</p>
  {% endif %}
</div>

{% if groups is not none %}
  {% if groups %}
  <form method="post" action="{{ url_for('dupes_resolve') }}">
    <div class="panel">
      <p><b>{{ groups|length }} grupo(s)</b> de duplicados — {{ surplus }} foto(s) sobrantes.
         Se movería lo tildado; las marcadas "conservar" quedan.</p>
      <button class="danger" onclick="return confirm('Se moverán las fotos tildadas a la carpeta _duplicados dentro del origen. ¿Continuar?')">
        🧹 Mover {{ surplus }} sobrante(s) a _duplicados
      </button>
    </div>
    {% for g in groups %}
    <div class="group">
      <div class="meta">Grupo de {{ g.files|length }} — {{ 'exactas' if g.exact else 'parecidas' }}</div>
      <div class="cards">
        {% for f in g.files %}
        <div class="thumb {% if f.keeper %}keeper{% endif %}">
          <a href="{{ url_for('photo', path=f.rel) }}" target="_blank" title="{{ f.rel }}">
            <img src="{{ url_for('dupe_thumb', idx=f.idx) }}" alt="foto"></a>
          <div>{{ f.w }}×{{ f.h }} · {{ (f.size/1024)|round|int }} KB</div>
          {% if f.keeper %}
            <span class="badge b-keep">conservar</span>
          {% else %}
            <span class="badge {{ 'b-exact' if f.exact else 'b-sim' }}">{{ 'exacta' if f.exact else 'parecida' }}</span>
            <div><label><input type="checkbox" name="rel" value="{{ f.rel }}" checked> mover</label></div>
          {% endif %}
        </div>
        {% endfor %}
      </div>
    </div>
    {% endfor %}
  </form>
  {% else %}
    <div class="panel">✅ No se encontraron duplicados.</div>
  {% endif %}
{% endif %}
</body>
</html>
"""


RENAME_TEMPLATE = """
<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>photo-sorter — renombrar</title>
<style>
  body { font-family: system-ui, sans-serif; margin: 2rem auto; max-width: 820px;
         padding: 0 1rem; background: #f5f5f5; }
  h1 { margin-bottom: .25rem; }
  .hint { color: #666; margin-bottom: 1.5rem; }
  .panel { background: #fff; border-radius: 8px; padding: 1rem; margin-bottom: 1rem;
           box-shadow: 0 1px 3px rgba(0,0,0,.1); }
  .row { display: flex; align-items: center; gap: .5rem; margin-bottom: .8rem; flex-wrap: wrap; }
  .row label { font-weight: 600; }
  input[type=text] { padding: .45rem; font-size: 1rem; }
  input.small { width: 70px; }
  .path { width: 380px; max-width: 80vw; }
  button { padding: .45rem .9rem; font-size: 1rem; cursor: pointer; }
  .primary { background: #1976d2; color: #fff; border: 0; border-radius: 4px; }
  .apply { background: #4caf50; color: #fff; border: 0; border-radius: 4px;
           padding: .6rem 1.2rem; font-size: 1.05rem; }
  button:disabled { opacity: .5; cursor: default; }
  .msg { background: #e8f5e9; border: 1px solid #4caf50; padding: .75rem 1rem;
         border-radius: 6px; margin-bottom: 1rem; }
  .err { background: #ffebee; border: 1px solid #e53935; padding: .75rem 1rem;
         border-radius: 6px; margin-bottom: 1rem; }
  .meta { color: #888; font-size: .85rem; }
  table { border-collapse: collapse; width: 100%; font-size: .92rem; }
  td, th { padding: .3rem .5rem; border-bottom: 1px solid #eee; text-align: left; }
  code { background: #f0f0f0; padding: 1px 5px; border-radius: 3px; }
  .example { font-size: 1.1rem; margin: .3rem 0; }
</style>
</head>
<body>
<h1>photo-sorter</h1>
<p><a href="{{ url_for('home') }}">Ordenar por rostro</a> · <a href="{{ url_for('dupes_page') }}">Buscar duplicados</a> · <b>Corregir carpeta</b></p>
<p class="hint">Renombra en masa las imágenes de una carpeta con el patrón que elijas: una palabra, un separador y un número que se va sumando solo (<code>0000</code>, <code>0001</code>…). Trabaja solo en el primer nivel de la carpeta (no entra en subcarpetas).</p>

{% if message %}<div class="msg">{{ message }}</div>{% endif %}
{% if error %}<div class="err">{{ error }}</div>{% endif %}

<div class="panel">
  <form method="post" action="{{ url_for('rename_preview') }}">
    <div class="row">
      <label>Carpeta:</label>
      <input class="path" type="text" name="folder" value="{{ f.folder }}" placeholder="Carpeta con las fotos a renombrar">
      <button type="button" onclick="browse('folder')">📂 Elegir…</button>
    </div>
    <div class="row">
      <label>Palabra:</label>
      <input type="text" name="prefix" value="{{ f.prefix }}" placeholder="ej: vacaciones">
      <label>Separador:</label>
      <input class="small" type="text" name="separator" value="{{ f.separator }}" placeholder="_">
      <label>Dígitos:</label>
      <input class="small" type="text" name="digits" value="{{ f.digits }}">
      <label>Empezar en:</label>
      <input class="small" type="text" name="start" value="{{ f.start }}">
    </div>
    <div class="row">
      <label>Orden:</label>
      <select name="order">
        <option value="name" {% if f.order=='name' %}selected{% endif %}>Por nombre</option>
        <option value="date" {% if f.order=='date' %}selected{% endif %}>Por fecha (más viejas primero)</option>
      </select>
      <button class="primary">👁 Vista previa</button>
    </div>
    <div class="example">Ejemplo: <code>{{ example }}</code></div>
  </form>
</div>

{% if plan is not none %}
<div class="panel">
  {% if plan %}
    <p><b>{{ total }} archivo(s)</b> se renombrarían así (muestro los primeros {{ plan|length }}):</p>
    <table>
      <tr><th>Actual</th><th>→</th><th>Nuevo</th></tr>
      {% for old, new in plan %}
      <tr><td>{{ old }}</td><td>→</td><td><b>{{ new }}</b></td></tr>
      {% endfor %}
    </table>
    <form method="post" action="{{ url_for('rename_apply') }}" style="margin-top:1rem">
      <input type="hidden" name="folder" value="{{ f.folder }}">
      <input type="hidden" name="prefix" value="{{ f.prefix }}">
      <input type="hidden" name="separator" value="{{ f.separator }}">
      <input type="hidden" name="digits" value="{{ f.digits }}">
      <input type="hidden" name="start" value="{{ f.start }}">
      <input type="hidden" name="order" value="{{ f.order }}">
      <button class="apply" onclick="return confirm('Se renombrarán {{ total }} archivo(s). ¿Continuar?')">
        ✏️ Renombrar {{ total }} archivo(s)
      </button>
      <span class="meta">Esto cambia los nombres reales en el disco.</span>
    </form>
  {% else %}
    <p>No hay imágenes en esa carpeta.</p>
  {% endif %}
</div>
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


def load_dupes():
    if not DUPES_PATH.is_file():
        return None
    with open(DUPES_PATH, encoding="utf-8") as f:
        return json.load(f)


def load_labels():
    if LABELS_PATH.is_file():
        with open(LABELS_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_labels(labels):
    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)


def _suspect_faces(members, enc):
    """Devuelve el set de face_ids que se despegan del centro del grupo (posible
    intruso). Adaptativo: marca los que superan la mediana del grupo + margen,
    con un piso absoluto. Solo para grupos de 3+ caras, con vectores."""
    if enc is None or len(members) < 3:
        return set()
    valid = [(fid, int(fid)) for fid in members if int(fid) < len(enc)]
    if len(valid) < 3:
        return set()
    vecs = enc[[i for _, i in valid]]
    cent = vecs.mean(axis=0)
    n = np.linalg.norm(cent)
    if n > 0:
        cent = cent / n
    dists = 1.0 - vecs @ cent
    thr = max(SUSPECT_FLOOR, float(np.median(dists)) + SUSPECT_MARGIN)
    return {fid for (fid, _), dist in zip(valid, dists) if dist > thr}


@app.route("/")
def home():
    cfg = load_config()
    data = load_clusters() if STATE["status"] != "running" else None
    labels = load_labels()

    enc = None
    if data and ENCODINGS_NPY.is_file():
        try:
            enc = np.load(ENCODINGS_NPY)
        except Exception:
            enc = None

    groups = []
    if data:
        faces = data["faces"]
        for c in data["clusters"]:
            members = c["faces"]
            suspects = _suspect_faces(members, enc)
            groups.append({
                "id": c["id"],
                "faces": [{"id": fid, "photo": faces[fid]["photo"],
                           "suspect": fid in suspects} for fid in members],
                "photos": len({faces[fid]["photo"] for fid in members}),
                "name": labels.get(str(c["id"]), ""),
                "suggestions": c.get("suggestions", []),  # candidatos ordenados (a confirmar)
                "review": c.get("review", False),  # grupo "Para revisar" (depurado)
                "suspects": len(suspects),
            })

    known = load_known()
    return render_template_string(
        TEMPLATE,
        cfg=cfg,
        state=STATE,
        imp=IMPORT_STATE,
        groups=groups,
        labeled=sum(1 for g in groups if g["name"]),
        known_count=len(known),
        known_dir=request.args.get("known_dir", ""),
        message=request.args.get("msg", ""),
    )


def _clear_analysis():
    """Borra el análisis anterior (grupos, nombres y miniaturas)."""
    CLUSTERS_PATH.unlink(missing_ok=True)
    LABELS_PATH.unlink(missing_ok=True)
    if FACES_DIR.is_dir():
        import shutil
        shutil.rmtree(FACES_DIR, ignore_errors=True)


def _save_folders(cfg):
    """Guarda origen/destino del formulario. Si cambió el origen, borra el
    análisis viejo así no quedan grupos de la carpeta anterior. Devuelve True
    si cambió el origen."""
    new_photos = request.form.get("photos_dir", "").strip()
    changed = new_photos != cfg.get("photos_dir", "")
    cfg["photos_dir"] = new_photos
    cfg["output_dir"] = request.form.get("output_dir", "").strip()
    save_config(cfg)
    if changed:
        _clear_analysis()
    return changed


@app.route("/settings", methods=["POST"])
def settings():
    cfg = load_config()
    _save_folders(cfg)
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
    def should_stop():
        return STATE.get("cancel", False)
    try:
        run_analysis(photos_dir, exclude=exclude, progress=progress, should_stop=should_stop)
        STATE["status"] = "stopped" if STATE.get("cancel") else "done"
    except Exception as e:
        STATE.update(status="error", error=str(e))


@app.route("/analyze", methods=["POST"])
def do_analyze():
    if STATE["status"] == "running":
        return redirect(url_for("home"))

    # Tomar las carpetas del formulario (lo que el usuario ve arriba) y guardarlas.
    cfg = load_config()
    _save_folders(cfg)

    photos_dir = Path(cfg.get("photos_dir", ""))
    if not cfg.get("photos_dir") or not photos_dir.is_dir():
        return redirect(url_for("home", msg="La carpeta de origen no existe. Elegila de nuevo."))

    exclude = Path(cfg["output_dir"]) if cfg.get("output_dir") else None
    STATE.update(status="running", current=0, total=0, photo="", error="", cancel=False)
    threading.Thread(target=_analysis_worker, args=(photos_dir, exclude), daemon=True).start()
    return redirect(url_for("home"))


@app.route("/analyze/stop", methods=["POST"])
def stop_analyze():
    if STATE["status"] == "running":
        STATE["cancel"] = True
    return redirect(url_for("home"))


def _import_worker(folder):
    def progress(current, total, name):
        IMPORT_STATE.update(current=current, total=total, photo=name)
    try:
        added = enroll_from_folder(folder, progress=progress)
        IMPORT_STATE.update(status="done", added=added)
    except Exception as e:
        IMPORT_STATE.update(status="error", error=str(e))


@app.route("/known/import", methods=["POST"])
def known_import():
    if IMPORT_STATE["status"] == "running":
        return redirect(url_for("home"))
    folder = request.form.get("known_dir", "").strip()
    if not folder or not Path(folder).is_dir():
        return redirect(url_for("home", msg="La carpeta a importar no existe.", known_dir=folder))
    IMPORT_STATE.update(status="running", current=0, total=0, photo="", error="")
    threading.Thread(target=_import_worker, args=(folder,), daemon=True).start()
    return redirect(url_for("home"))


@app.route("/known/clear", methods=["POST"])
def known_clear():
    KNOWN_PATH.unlink(missing_ok=True)
    if KNOWN_THUMBS.is_dir():
        import shutil
        shutil.rmtree(KNOWN_THUMBS, ignore_errors=True)
    return redirect(url_for("home", msg="Base de personas conocidas vaciada."))


KNOWN_TEMPLATE = """
<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>photo-sorter — personas conocidas</title>
<style>
  body { font-family: system-ui, sans-serif; margin: 2rem auto; max-width: 900px;
         padding: 0 1rem; background: #f5f5f5; }
  h1 { margin-bottom: .25rem; }
  .hint { color: #666; margin-bottom: 1.5rem; }
  .grid { display: flex; flex-wrap: wrap; gap: 14px; }
  .person { background: #fff; border-radius: 8px; padding: .6rem; width: 150px;
            box-shadow: 0 1px 3px rgba(0,0,0,.1); text-align: center; }
  .person img { width: 130px; height: 130px; object-fit: cover; border-radius: 6px; }
  .noimg { width: 130px; height: 130px; border-radius: 6px; background: #e0e0e0;
           display: flex; align-items: center; justify-content: center; font-size: 2.5rem; color:#aaa; }
  .pname { font-weight: 600; margin-top: .35rem; word-break: break-word; }
  .pcount { color: #888; font-size: .8rem; }
  button { padding: .3rem .6rem; font-size: .85rem; cursor: pointer;
           background: #eee; border: 1px solid #ccc; border-radius: 4px; margin-top: .3rem; }
  .msg { background: #e8f5e9; border: 1px solid #4caf50; padding: .75rem 1rem;
         border-radius: 6px; margin-bottom: 1rem; }
</style>
</head>
<body>
<h1>photo-sorter</h1>
<p><a href="{{ url_for('home') }}">Ordenar por rostro</a> · <a href="{{ url_for('dupes_page') }}">Buscar duplicados</a> · <a href="{{ url_for('rename_page') }}">Corregir carpeta</a> · <b>Personas conocidas</b></p>
<p class="hint">Estas son las personas que el programa ya reconoce solo. La miniatura es la cara más clara que aprendió de cada una.</p>
{% if message %}<div class="msg">{{ message }}</div>{% endif %}

{% if people %}
<p><b>{{ people|length }}</b> persona(s) conocida(s):</p>
<div class="grid">
  {% for p in people %}
  <div class="person">
    {% if p.thumb %}
      <img src="{{ url_for('known_thumb', name=p.name) }}" alt="{{ p.name }}">
    {% else %}
      <div class="noimg">👤</div>
    {% endif %}
    <div class="pname">{{ p.name }}</div>
    <div class="pcount">{{ p.count }} cara(s) aprendida(s)</div>
    <form method="post" action="{{ url_for('known_forget') }}"
          onsubmit="return confirm('¿Olvidar a {{ p.name }}? Se dejará de reconocer sola.')">
      <input type="hidden" name="name" value="{{ p.name }}">
      <button>🗑 Olvidar</button>
    </form>
  </div>
  {% endfor %}
</div>
{% else %}
<div class="person" style="width:auto">Todavía no hay personas conocidas. Importá una carpeta ordenada o nombrá gente al analizar.</div>
{% endif %}
</body>
</html>
"""


@app.route("/known")
def known_page():
    return render_template_string(
        KNOWN_TEMPLATE, people=known_summary(), message=request.args.get("msg", ""))


@app.route("/known_thumb/<name>")
def known_thumb(name):
    return send_from_directory(KNOWN_THUMBS, f"{safe_name(name)}.jpg")


@app.route("/known/forget", methods=["POST"])
def known_forget():
    name = request.form.get("name", "")
    if name:
        forget_person(name)
    return redirect(url_for("known_page", msg=f"Se olvidó a {name}."))


@app.route("/cluster/discard", methods=["POST"])
def cluster_discard():
    """Descarta un grupo entero: son detecciones que NO son personas (tapizado,
    manchas, objetos que parecen caras). Se saca del análisis; no se organiza ni
    se vuelve a preguntar. No borra fotos del disco."""
    data = load_clusters()
    if not data:
        return {"ok": False}
    cid = str(request.form.get("cluster_id", ""))
    data["clusters"] = [c for c in data["clusters"] if str(c["id"]) != cid]
    labels = load_labels()
    if labels.pop(cid, None) is not None:
        save_labels(labels)
    with open(CLUSTERS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return {"ok": True}


@app.route("/cluster/remove", methods=["POST"])
def cluster_remove():
    """Saca caras de un grupo (que no son esa persona) y las manda a un grupo
    aparte 'Para revisar', para una segunda pasada. No borra fotos del disco."""
    data = load_clusters()
    if not data:
        return {"ok": False}
    cluster_id = str(request.form.get("cluster_id", ""))
    face_ids = [f for f in request.form.get("face_ids", "").split(",") if f]
    if not face_ids:
        return {"ok": True, "removed": 0}

    src = next((c for c in data["clusters"] if str(c["id"]) == cluster_id), None)
    if not src:
        return {"ok": False}

    to_move = [f for f in src["faces"] if f in set(face_ids)]
    src["faces"] = [f for f in src["faces"] if f not in set(face_ids)]

    # juntar todo lo apartado en un único grupo "Para revisar" (o crearlo)
    review = next((c for c in data["clusters"] if c.get("review")), None)
    if review is None:
        new_id = max((c["id"] for c in data["clusters"]), default=-1) + 1
        review = {"id": new_id, "faces": [], "review": True}
        data["clusters"].append(review)
    review["faces"].extend(to_move)

    # si el grupo de origen quedó vacío, sacarlo (y su etiqueta)
    if not src["faces"]:
        data["clusters"] = [c for c in data["clusters"] if c is not src]
        labels = load_labels()
        if labels.pop(cluster_id, None) is not None:
            save_labels(labels)

    with open(CLUSTERS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return {"ok": True, "removed": len(to_move)}


@app.route("/label", methods=["POST"])
def label():
    labels = load_labels()
    cluster_id = request.form["cluster_id"]
    name = request.form["name"].strip()
    if name:
        labels[cluster_id] = name
        _enroll_cluster(cluster_id, name)  # sumar esta persona a la base conocida
    else:
        labels.pop(cluster_id, None)
    save_labels(labels)
    return {"ok": True, "name": name}


def _enroll_cluster(cluster_id, name):
    """Suma las caras de un grupo recién nombrado a la base de personas y, si la
    persona no tiene miniatura, guarda la de la cara más grande del grupo."""
    data = load_clusters()
    if not data or not ENCODINGS_NPY.is_file():
        return
    cluster = next((c for c in data["clusters"] if str(c["id"]) == str(cluster_id)), None)
    if not cluster:
        return
    try:
        enc = np.load(ENCODINGS_NPY)
    except Exception:
        return
    idxs = [int(fid) for fid in cluster["faces"] if int(fid) < len(enc)]
    if not idxs:
        return
    db = load_known()
    add_faces(db, name, [enc[i] for i in idxs])
    save_known(db)

    # miniatura: la cara más grande del grupo (si la persona aún no tiene una)
    thumb_path = KNOWN_THUMBS / f"{safe_name(name)}.jpg"
    if not thumb_path.is_file():
        faces = data["faces"]
        def area(fid):
            t, r, b, l = faces[fid]["box"]
            return (b - t) * (r - l)
        biggest = max(cluster["faces"], key=area)
        src = FACES_DIR / f"{biggest}.jpg"
        if src.is_file():
            KNOWN_THUMBS.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy2(src, thumb_path)


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


@app.route("/dupes")
def dupes_page():
    cfg = load_config()
    data = load_dupes() if DUPE_STATE["status"] != "running" else None
    return render_template_string(
        DUPES_TEMPLATE,
        folder=cfg.get("photos_dir", ""),
        state=DUPE_STATE,
        groups=data["groups"] if data else None,
        surplus=sum(len(g["files"]) - 1 for g in data["groups"]) if data else 0,
        message=request.args.get("msg", ""),
    )


def _dupes_worker(folder, threshold, exclude):
    def progress(current, total, name):
        DUPE_STATE.update(current=current, total=total, photo=name)
    try:
        find_duplicates(folder, threshold=threshold, exclude=exclude, progress=progress)
        DUPE_STATE["status"] = "done"
    except Exception as e:
        DUPE_STATE.update(status="error", error=str(e))


@app.route("/dupes/scan", methods=["POST"])
def dupes_scan():
    if DUPE_STATE["status"] == "running":
        return redirect(url_for("dupes_page"))
    cfg = load_config()
    folder = Path(cfg.get("photos_dir", ""))
    if not cfg.get("photos_dir") or not folder.is_dir():
        return redirect(url_for("dupes_page", msg="Elegí primero la carpeta de origen (pestaña de rostros)."))
    exclude = Path(cfg["output_dir"]) if cfg.get("output_dir") else None
    DUPE_STATE.update(status="running", current=0, total=0, photo="", error="")
    threading.Thread(target=_dupes_worker, args=(folder, DEFAULT_THRESHOLD, exclude), daemon=True).start()
    return redirect(url_for("dupes_page"))


@app.route("/dupes/resolve", methods=["POST"])
def dupes_resolve():
    rels = request.form.getlist("rel")
    if not rels:
        return redirect(url_for("dupes_page", msg="No marcaste ninguna foto para mover."))
    moved, dupes_dir, errors = resolve_duplicates(rels)

    # Quitar del índice las fotos movidas (sin re-escanear todo); los grupos
    # que quedan con una sola foto ya no son duplicados.
    data = load_dupes()
    if data:
        gone = set(rels)
        new_groups = []
        for g in data["groups"]:
            keep = [f for f in g["files"] if f["rel"] not in gone]
            if len(keep) >= 2:
                new_groups.append({"files": keep, "exact": all(f["exact"] for f in keep)})
        data["groups"] = new_groups
        with open(DUPES_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    msg = f"Listo: {moved} foto(s) movidas a {dupes_dir}."
    if errors:
        msg += f" {len(errors)} no se pudieron mover."
    return redirect(url_for("dupes_page", msg=msg))


@app.route("/dupe_thumb/<int:idx>")
def dupe_thumb(idx):
    return send_from_directory(DUPES_THUMBS, f"{idx:06d}.jpg")


PREVIEW_LIMIT = 30


def _rename_form(overrides=None):
    cfg = load_config()
    form = {
        "folder": cfg.get("photos_dir", ""),
        "prefix": "foto",
        "separator": "_",
        "digits": "4",
        "start": "1",
        "order": "name",
    }
    if overrides:
        form.update({k: v for k, v in overrides.items() if v is not None})
    return form


def _rename_example(form):
    try:
        digits = max(1, min(int(form["digits"]), 12))
    except (ValueError, KeyError):
        digits = 4
    try:
        start = int(form["start"])
    except (ValueError, KeyError):
        start = 1
    prefix = "".join(c for c in form["prefix"] if c not in '<>:"/\\|?*')
    return f"{prefix}{form['separator']}{str(start).zfill(digits)}.jpg"


@app.route("/rename")
def rename_page():
    form = _rename_form()
    return render_template_string(
        RENAME_TEMPLATE, f=form, example=_rename_example(form),
        plan=None, total=0, message=request.args.get("msg", ""), error="")


@app.route("/rename/preview", methods=["POST"])
def rename_preview():
    form = _rename_form(request.form.to_dict())
    error = ""
    plan_full = []
    folder = Path(form["folder"])
    if not form["folder"] or not folder.is_dir():
        error = "La carpeta no existe. Elegila de nuevo."
    elif not form["prefix"].strip():
        error = "Escribí la palabra (prefijo) para los nombres."
    else:
        plan_full = plan_rename(folder, form["prefix"], form["digits"],
                                separator=form["separator"],
                                start=int(form["start"]) if form["start"].lstrip("-").isdigit() else 1,
                                order=form["order"])
    return render_template_string(
        RENAME_TEMPLATE, f=form, example=_rename_example(form),
        plan=(plan_full[:PREVIEW_LIMIT] if not error else None),
        total=len(plan_full), message="", error=error)


@app.route("/rename/apply", methods=["POST"])
def rename_apply():
    form = _rename_form(request.form.to_dict())
    folder = Path(form["folder"])
    if not form["folder"] or not folder.is_dir():
        return redirect(url_for("rename_page", msg="La carpeta no existe."))
    renamed, errors = apply_rename(
        folder, form["prefix"], form["digits"],
        separator=form["separator"],
        start=int(form["start"]) if form["start"].lstrip("-").isdigit() else 1,
        order=form["order"])
    msg = f"Listo: {renamed} archivo(s) renombrados."
    if errors:
        msg += f" {len(errors)} con problemas (ej: {errors[0][0]})."
    return redirect(url_for("rename_page", msg=msg))


@app.route("/face/<face_id>")
def face(face_id):
    return send_from_directory(FACES_DIR, f"{face_id}.jpg")


@app.route("/photo/<path:path>")
def photo(path):
    # sirve la foto original desde el origen (analisis de rostros o duplicados)
    data = load_clusters()
    base = None
    if data:
        base = Path(data["photos_dir"])
    if base is None or not (base / path).is_file():
        cfg = load_config()
        if cfg.get("photos_dir"):
            base = Path(cfg["photos_dir"])
    return send_from_directory(base, path)


def main():
    print("[web] Abrí http://127.0.0.1:5000 en el navegador (solo accesible desde esta máquina).")
    threading.Timer(1.5, lambda: webbrowser.open("http://127.0.0.1:5000")).start()
    app.run(host="127.0.0.1", port=5000, debug=False)


if __name__ == "__main__":
    main()
