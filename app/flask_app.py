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

import hashlib
import json
import math
import threading
import time
import unicodedata
import webbrowser
from pathlib import Path

from flask import (Flask, abort, redirect, render_template_string, request,
                   send_from_directory, url_for)

import numpy as np
from PIL import Image, ImageOps

from app import (CLUSTERS_PATH, CORRECT_THUMBS, DUPES_PATH, DUPES_THUMBS,
                 ENCODINGS_NPY, FACES_DIR, FIND_THUMBS, GALLERY_THUMBS, GROUP_DIR_NAME,
                 KNOWN_PATH, KNOWN_THUMBS, LABELS_PATH, PICK_THUMBS)
from app.analyze import MIN_FACE_PX, run_analysis
from app.config import get_library, load_config, save_config, set_library
from app.corrector import apply_correction, scan_person_folder
from app.duplicates import DEFAULT_THRESHOLD, find_duplicates, resolve_duplicates
from app.finder import (EPS_DEFAULT, EPS_MAX, EPS_MIN, find_known, find_person,
                        move_to_person)
from app.known import (add_faces, enroll_from_folder, forget_person, known_summary,
                       load_known, move_photo, person_photos, save_face_crops,
                       save_known, safe_name, set_cover)
from app.linkgrab import grab_links
from app.notfaces import add_not_faces, clear_not_faces, count_not_faces
from app.nsfw import (NSFW_DIR_NAME, NSFW_THUMBS, SCORE_MIN, SCORE_MIN_HI,
                      SCORE_MIN_LO, move_to_nsfw, scan_nsfw)
from app import person_meta
from app import tags as tags_mod
from app import interrogator, vlm
from app import posts as posts_mod
from app import phrases as phrases_mod
from app.organizer import organize
from app.renamer import apply_rename, plan_rename
from app.splitter import SPLIT_THUMBS, move_to_group, scan_groups
from app.stats import archive_current, record_verdict, summary as stats_summary

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
# Estado del buscador de persona (otro hilo); los resultados viven acá hasta
# la próxima búsqueda (no se guardan en disco, salvo las miniaturas)
FIND_STATE = {"status": "idle", "current": 0, "total": 0, "photo": "", "error": "",
              "results": [], "by_person": [], "mode": "one", "name": "", "folder": "",
              "eps": EPS_DEFAULT, "scanned": 0, "media": "images", "cleared": False}
# Estado del corrector de entrenamiento (otro hilo); "data" guarda el escaneo
# clasificado (con los vectores) hasta que el usuario aplica la corrección
CORRECT_STATE = {"status": "idle", "current": 0, "total": 0, "photo": "",
                 "error": "", "name": "", "folder": "", "data": None}
# Estado de la revisión NSFW (otro hilo)
NSFW_STATE = {"status": "idle", "current": 0, "total": 0, "photo": "", "error": "",
              "results": [], "folder": "", "scanned": 0,
              "score_min": SCORE_MIN, "covered": False}
# Estado del separador por cantidad de caras (otro hilo)
SPLIT_STATE = {"status": "idle", "current": 0, "total": 0, "photo": "", "error": "",
               "results": [], "folder": "", "min_px": MIN_FACE_PX, "resumen": {}}


# ------------------------------------------------------------------------
# Tema visual compartido — "Cuarto oscuro": barra lateral de íconos, fondo
# casi negro, acento azul frío. Un solo CSS y una sola navegación para todas
# las páginas (antes cada página repetía sus estilos y su barra de enlaces).
# ------------------------------------------------------------------------

BASE_CSS = """
  :root {
    --bg:#0c0c0e; --side:#131319; --panel:#181920; --panel2:#20222b;
    --bd:#2c2e38; --tx:#e7e8ec; --mut:#8b8d99; --acc:#5b8fd0; --acc2:#79a7df;
    --grn:#3f9d5a; --red:#d6564e; --amb:#d99a3a; --viol:#9b6cc4;
  }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--tx);
         font-family:"Instrument Sans","Segoe UI",system-ui,sans-serif; }
  a { color:var(--acc2); text-decoration:none; }
  a:hover { color:#c3d6ec; text-decoration:underline; }
  h1 { margin:.2rem 0 .5rem; font-size:1.5rem; font-weight:650; }
  code { background:#000; color:#cbd6e6; padding:1px 5px; border-radius:3px; font-size:.9em; }
  .hint { color:var(--mut); margin-bottom:1.5rem; max-width:70ch; }
  .meta { color:var(--mut); font-size:.85rem; }
  .counter { font-size:1.1rem; font-weight:650; margin-bottom:.5rem; }
  .done-big { font-size:1.3rem; margin:.5rem 0; }

  /* layout: barra lateral + contenido */
  .app { display:flex; min-height:100vh; }
  .sidebar { width:212px; flex-shrink:0; background:var(--side);
             border-right:1px solid var(--bd); padding:1rem .6rem; position:sticky;
             top:0; height:100vh; overflow-y:auto; }
  .sidebar .brand { font-weight:700; font-size:1.05rem; letter-spacing:.02em;
                    padding:.4rem .7rem 1rem; color:var(--tx); }
  .sidebar .brand small { display:block; color:var(--mut); font-weight:400; font-size:.7rem; letter-spacing:.06em; }
  .sidebar a { display:flex; align-items:center; gap:.7rem; padding:.55rem .7rem;
               border-radius:8px; color:var(--mut); font-size:.95rem; margin-bottom:2px; }
  .sidebar a:hover { background:var(--panel2); color:var(--tx); text-decoration:none; }
  .sidebar a .ic { width:1.4rem; text-align:center; font-size:1.1rem; }
  .sidebar a.active { background:var(--panel2); color:var(--tx);
                      box-shadow:inset 3px 0 0 var(--acc); }
  .main { flex:1; padding:2rem 2.6rem; max-width:1040px; min-width:0; }
  @media (max-width:720px) {
    .app { flex-direction:column; }
    .sidebar { width:auto; height:auto; position:static; display:flex;
               flex-wrap:wrap; gap:2px; }
    .sidebar .brand { width:100%; padding-bottom:.4rem; }
    .main { padding:1.2rem; }
  }

  /* tarjetas / paneles */
  .panel, .card, .group { background:var(--panel); border:1px solid var(--bd);
    border-radius:10px; padding:1rem; margin-bottom:1rem; }

  /* filas de formulario */
  .row { display:flex; align-items:center; gap:.5rem; margin-bottom:.6rem; flex-wrap:wrap; }
  .row label { font-weight:600; }

  /* controles */
  input[type=text], select, textarea {
    padding:.45rem .55rem; font-size:1rem; background:var(--panel2);
    color:var(--tx); border:1px solid var(--bd); border-radius:6px; }
  input[type=text]::placeholder { color:#6a6c78; }
  input.small { width:70px; }
  .path { width:420px; max-width:85vw; }
  input[type=range] { width:220px; accent-color:var(--acc); }
  input[type=checkbox], input[type=radio] { accent-color:var(--acc); }
  progress { width:100%; height:18px; }
  progress::-webkit-progress-bar { background:var(--panel2); border-radius:4px; }
  progress::-webkit-progress-value { background:var(--acc); border-radius:4px; }

  /* botones */
  button { padding:.45rem .9rem; font-size:1rem; cursor:pointer; border-radius:6px;
    border:1px solid var(--bd); background:var(--panel2); color:var(--tx); }
  button:hover { border-color:#3c3e4a; }
  button:disabled { opacity:.45; cursor:default; }
  .primary { background:var(--acc); color:#0b1420; border:0; font-weight:600; }
  .primary:hover { background:var(--acc2); }
  .secondary { background:var(--panel2); color:var(--tx); }
  .organize, .apply { background:var(--grn); color:#07130c; border:0; font-weight:600;
    padding:.6rem 1.2rem; font-size:1.05rem; }
  .danger { background:var(--red); color:#fff; border:0; }
  .dismiss { float:right; background:none; border:0; font-size:1rem; cursor:pointer;
    color:inherit; opacity:.55; padding:0 .2rem; }
  .dismiss:hover { opacity:1; }

  /* avisos */
  .msg { background:rgba(63,157,90,.13); border:1px solid var(--grn); color:#bfe6cb;
    padding:.75rem 1rem; border-radius:8px; margin-bottom:1rem; }
  .err { background:rgba(214,86,78,.13); border:1px solid var(--red); color:#f0c3bf;
    padding:.75rem 1rem; border-radius:8px; margin-bottom:1rem; }

  /* rostros / grupos (home + modal) */
  .faces { display:flex; flex-wrap:wrap; gap:6px; margin:.75rem 0; }
  .faces a img { height:110px; border-radius:5px; display:block; }
  .facewrap { position:relative; display:inline-block; }
  .facewrap .fx { position:absolute; top:3px; right:3px; background:var(--red);
    color:#fff; border-radius:50%; width:22px; height:22px; line-height:22px;
    text-align:center; font-size:.8rem; cursor:pointer; opacity:.85; user-select:none; }
  .facewrap .fx:hover { opacity:1; }
  .facewrap.suspect a img { outline:3px solid var(--amb); }
  .facewrap.suspect .warn { position:absolute; top:3px; left:3px; background:var(--amb);
    color:#000; border-radius:4px; font-size:.7rem; padding:0 4px; }
  .suspect-note { color:var(--amb); font-weight:600; }
  .suggest-note { color:var(--acc2); margin:.3rem 0; }
  .ask { font-size:1.15rem; margin-right:.3rem; }
  .ask-cover { height:110px; border-radius:6px; outline:3px solid var(--acc); }
  #wizard-card { transition:opacity .25s, transform .25s; }
  #wizard-card.fade { opacity:0; transform:translateY(-10px); }
  #list-view .card.labeled { border-left:5px solid var(--grn); }
  .saved-tick { color:#7fd39a; font-weight:600; margin-left:.5rem; opacity:0; transition:opacity .3s; }
  .saved-tick.show { opacity:1; }

  /* modal depurar grupo */
  #modal-bg { display:none; position:fixed; inset:0; background:rgba(0,0,0,.7); z-index:50; }
  #modal { background:var(--panel); border:1px solid var(--bd); border-radius:10px;
    max-width:900px; width:92vw; margin:3vh auto; max-height:94vh; display:flex; flex-direction:column; }
  #modal-head { padding:1rem; border-bottom:1px solid var(--bd); }
  #modal-body { padding:1rem; overflow-y:auto; }
  #modal-foot { padding:1rem; border-top:1px solid var(--bd); display:flex; gap:.5rem;
    align-items:center; flex-wrap:wrap; }
  .pick { display:inline-block; position:relative; cursor:pointer; }
  .pick img { height:120px; border-radius:5px; display:block; border:3px solid transparent; }
  .pick.sel img { border-color:var(--red); }
  .pick .x { position:absolute; top:4px; right:4px; background:var(--red); color:#fff;
    border-radius:50%; width:22px; height:22px; text-align:center; line-height:22px; font-size:.8rem; opacity:0; }
  .pick.sel .x { opacity:1; }
  .pick.suspectpick img { outline:3px solid var(--amb); }
  .pick .warn2 { position:absolute; top:4px; left:4px; background:var(--amb); color:#000;
    font-size:.7rem; padding:0 4px; border-radius:4px; }
  .pick .pn { font-size:.7rem; color:var(--mut); text-align:center; max-width:120px;
    overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }

  /* duplicados */
  .cards { display:flex; flex-wrap:wrap; gap:12px; }
  .thumb { text-align:center; font-size:.8rem; }
  .thumb img { max-height:160px; max-width:200px; border-radius:5px; display:block; }
  .thumb.keeper img { outline:4px solid var(--grn); }
  .badge { display:inline-block; padding:1px 6px; border-radius:10px; color:#fff; font-size:.72rem; }
  .b-keep { background:var(--grn); color:#07130c; } .b-exact { background:var(--viol); } .b-sim { background:var(--amb); color:#000; }

  /* renombrar */
  table { border-collapse:collapse; width:100%; font-size:.92rem; }
  td, th { padding:.3rem .5rem; border-bottom:1px solid var(--bd); text-align:left; }
  .example { font-size:1.1rem; margin:.3rem 0; }

  /* galería */
  .grid { display:flex; flex-wrap:wrap; gap:14px; }
  .person { background:var(--panel); border:1px solid var(--bd); border-radius:10px;
    padding:.6rem; width:150px; text-align:center; }
  .person a { color:inherit; text-decoration:none; }
  .person a:hover img, .person a:hover .noimg { outline:3px solid var(--acc); }
  .person img { width:130px; height:130px; object-fit:cover; border-radius:8px; }
  .noimg { width:130px; height:130px; border-radius:8px; background:var(--panel2);
    display:flex; align-items:center; justify-content:center; font-size:2.5rem; color:var(--mut); }
  .pname { font-weight:600; margin-top:.35rem; word-break:break-word; }
  .pcount { color:var(--mut); font-size:.8rem; }

  /* página de persona */
  .cover-box { display:flex; align-items:center; gap:1rem; }
  .cover-box img { height:150px; border-radius:8px; }
  /* galería del perfil: masonry estilo VSCO — cada foto conserva su forma
     (vertical alta, apaisada baja) y se encolumnan solas por ancho */
  .masonry { column-width:220px; column-gap:12px; }
  @media (max-width:640px) { .masonry { column-width:150px; column-gap:8px; } }
  .masonry .shot { break-inside:avoid; -webkit-column-break-inside:avoid;
    background:var(--panel); border:1px solid var(--bd); border-radius:10px;
    padding:.4rem; margin:0 0 12px; text-align:center; }
  .masonry .shot img { width:100%; height:auto; display:block; border-radius:8px;
    cursor:zoom-in; }
  .shot .fn { font-size:.72rem; color:var(--mut); margin:.25rem 0 .35rem; overflow:hidden;
    text-overflow:ellipsis; white-space:nowrap; }
  .shot .primary { width:100%; }
  .opt { text-align:center; }
  .opt img { height:110px; border-radius:6px; display:block; margin-bottom:.25rem;
    border:3px solid transparent; cursor:zoom-in; }
  .opt.sug img { border-color:var(--grn); }
  .opt input:checked + img { outline:3px solid var(--acc); }
  .opt.none input:checked ~ span, .opt.none input:checked + span { outline:3px solid var(--acc); }
  .none-box { display:inline-block; padding:.6rem; border-radius:8px; background:rgba(214,86,78,.14); }
  .stay-box { display:inline-block; padding:.6rem; border-radius:8px; background:rgba(217,154,58,.14); }

  /* acordeón de personas encontradas (buscador general) */
  details.grp > summary { cursor:pointer; list-style:none; padding:.15rem 0;
    display:flex; align-items:center; gap:.5rem; user-select:none; }
  details.grp > summary::-webkit-details-marker { display:none; }
  details.grp > summary::before { content:"▾"; color:var(--acc); font-size:.9rem;
    transition:transform .15s; }
  details.grp:not([open]) > summary::before { transform:rotate(-90deg); }
  details.grp:not([open]) > summary { color:var(--mut); }
  details.grp > summary:hover { color:var(--acc2); }

  /* resultados buscador / nsfw */
  .hit { background:var(--panel); border:1px solid var(--bd); border-radius:10px;
    padding:.5rem; width:184px; text-align:center; }
  .hit img { max-height:150px; max-width:168px; border-radius:5px; cursor:zoom-in; }
  .hit .fn { font-size:.72rem; color:var(--mut); margin:.2rem 0; overflow:hidden;
    text-overflow:ellipsis; white-space:nowrap; }
  .hit .cf { font-size:.8rem; color:var(--acc2); font-weight:600; }
  .hit .cf.nsfw { color:#e78; }

  /* enlaces */
  ol { margin:.5rem 0; padding-left:2rem; }
  ol li { margin-bottom:.25rem; word-break:break-all; }

  /* etiquetas (chips) */
  .chips { display:flex; flex-wrap:wrap; gap:6px; align-items:center; }
  .chip { display:inline-flex; align-items:center; gap:.35rem; background:var(--panel2);
    border:1px solid var(--bd); border-radius:20px; padding:.2rem .6rem; font-size:.85rem; }
  .chip .x { cursor:pointer; opacity:.6; font-weight:700; }
  .chip .x:hover { opacity:1; color:var(--red); }
  .chip.inherited { opacity:.75; border-style:dashed; }
  .taginput { min-width:150px; border:0; background:transparent; color:var(--tx);
    font-size:.9rem; padding:.2rem; outline:none; }
  .shot .phototags { display:flex; flex-wrap:wrap; gap:3px; margin-top:.25rem; }
  .shot .phototags .chip { font-size:.7rem; padding:.05rem .45rem; }
  .shot .ptadd { width:100%; font-size:.75rem; margin-top:.2rem; padding:.2rem .3rem; }
  /* etiquetador una-por-una: foto a la izquierda, etiquetas a la derecha */

  /* botones-atajo del etiquetador */
  .sc { border-radius:20px; padding:.25rem .7rem; font-size:.85rem; position:relative; }
  .sc.active { background:var(--acc); color:#0b1420; border-color:var(--acc); font-weight:600; }
  .sc .scx { margin-left:.4rem; opacity:.6; cursor:pointer; }
  .sc .scx:hover { opacity:1; color:var(--red); }

  /* modal foto completa (persona/corrector/nsfw) */
  #fmodal-bg { display:none; position:fixed; inset:0; background:rgba(0,0,0,.8);
    z-index:50; align-items:center; justify-content:center; }
  #fmodal-bg img { max-width:92vw; max-height:88vh; border-radius:8px; }
  #fmodal-x { position:fixed; top:14px; right:22px; color:#fff; font-size:2rem;
    cursor:pointer; user-select:none; }

  /* lightbox de galería: zoom con ruedita + arrastre + flechas */
  #lb { display:none; position:fixed; inset:0; background:rgba(0,0,0,.92); z-index:60;
    overflow:hidden; touch-action:none; }
  #lb .stage { position:absolute; inset:0; z-index:1; display:flex; align-items:center; justify-content:center; }
  #lb img { max-width:94vw; max-height:92vh; user-select:none; -webkit-user-drag:none;
    transform-origin:center center; will-change:transform; cursor:grab; }
  #lb img.grabbing { cursor:grabbing; }
  /* z-index:2 para que flechas y ✕ queden POR ENCIMA del fondo (.stage), si no
     el fondo intercepta el clic y cierra en vez de pasar de foto */
  #lb .lb-btn { position:fixed; top:50%; transform:translateY(-50%); font-size:2.6rem;
    color:#fff; background:rgba(0,0,0,.35); border:0; width:2.4rem; height:3.4rem;
    border-radius:8px; cursor:pointer; opacity:.75; z-index:2; }
  #lb .lb-btn:hover { opacity:1; }
  #lb #lb-prev { left:18px; } #lb #lb-next { right:18px; }
  #lb #lb-x { position:fixed; top:14px; right:20px; font-size:2rem; color:#fff;
    cursor:pointer; user-select:none; opacity:.8; z-index:2; }
  #lb #lb-x:hover { opacity:1; }
  #lb .lb-cap { position:fixed; bottom:14px; left:0; right:0; text-align:center;
    color:#cfd2da; font-size:.85rem; pointer-events:none; }
  #lb .lb-hint { position:fixed; top:16px; left:20px; color:#8b8d99; font-size:.8rem; }

  /* --- visor estilo publicación (solo la galería de una persona) --- */
  /* foto a la izquierda, columna del post a la derecha */
  #lb.lb-post .post { position:absolute; inset:0; display:grid;
    grid-template-columns:minmax(0,1fr) 380px; }
  #lb.lb-post .stage { position:relative; inset:auto; overflow:hidden; }
  #lb.lb-post img { max-width:100%; max-height:100%; }
  #lb.lb-post .side { background:var(--panel); border-left:1px solid var(--line);
    display:flex; flex-direction:column; min-height:0; }
  #lb.lb-post .phead { display:flex; align-items:center; gap:.6rem; padding:.7rem .9rem;
    border-bottom:1px solid var(--line); flex:0 0 auto; }
  #lb.lb-post .av { width:38px; height:38px; border-radius:50%; object-fit:cover; }
  #lb.lb-post .phead .meta { font-size:.72rem; word-break:break-all; }
  /* el cuerpo scrollea; el perfil y el pie de la columna quedan fijos */
  #lb.lb-post .pbody { flex:1 1 auto; overflow-y:auto; padding:.8rem .9rem; min-height:0; }
  #lb.lb-post .pcap { margin-bottom:1rem; }
  #lb.lb-post .pcap p { margin:0; line-height:1.5; white-space:pre-wrap; }
  #lb.lb-post .empty { color:var(--dim); font-style:italic; }
  #lb.lb-post .capin { width:100%; margin-top:.4rem; min-height:5rem; resize:vertical;
    background:var(--bg); color:var(--fg); border:1px solid var(--line);
    border-radius:6px; padding:.4rem; font:inherit; font-size:.85rem; }
  #lb.lb-post .com { padding:.3rem 0; font-size:.88rem; line-height:1.4;
    border-bottom:1px solid var(--line); }
  #lb.lb-post .com b { color:var(--fg); }
  #lb.lb-post .ptags { flex:0 0 auto; padding:.6rem .9rem; border-top:1px solid var(--line);
    max-height:45vh; overflow-y:auto; }
  #lb.lb-post .pfoot { flex:0 0 auto; display:flex; justify-content:space-between;
    align-items:center; gap:.5rem; padding:.5rem .9rem; border-top:1px solid var(--line); }
  #lb.lb-post .pfoot select { font-size:.75rem; max-width:230px; }
  #lb.lb-post .sm { padding:.15rem .5rem; font-size:.78rem; }
  #lb.lb-post .sm.on { border-color:var(--accent); color:var(--accent); }
  /* las flechas no se pueden ir sobre la columna del post */
  #lb.lb-post #lb-next { right:calc(380px + 18px); }
  #lb.lb-post #lb-x { right:calc(380px + 20px); }
  /* pantalla angosta: el post pasa a estar abajo de la foto */
  @media (max-width:900px) {
    #lb.lb-post .post { grid-template-columns:1fr; grid-template-rows:50vh 1fr; }
    #lb.lb-post .side { border-left:0; border-top:1px solid var(--line); }
    #lb.lb-post #lb-next { right:18px; }
    #lb.lb-post #lb-x { right:20px; }
  }
"""

# Cuántas fotos por página en la galería de una persona. Sin paginar, 1500 fotos
# son 1500 <select> con toda la gente adentro y el navegador se arrastra.
PER_OPTIONS = [10, 20, 50, 100]
PER_DEFAULT = 50

# (ícono, etiqueta, endpoint Flask, clave de página activa)
NAV_ITEMS = [
    ("▦", "Rostros", "home", "home"),
    ("⧉", "Duplicados", "dupes_page", "dupes"),
    ("✎", "Carpeta", "rename_page", "rename"),
    ("⛓", "Enlaces", "links_page", "links"),
    ("☻", "Galería", "known_page", "known"),
    ("⌕", "Persona", "finder_page", "finder"),
    ("⚠", "NSFW", "nsfw_page", "nsfw"),
    ("⚇", "Separar caras", "split_page", "split"),
    ("✎", "Frases", "phrases_page", "phrases"),
    ("⚙", "Ajustes", "settings_page", "settings"),
]


def _nav_html(active):
    """Barra lateral común; resalta la página activa (clave `active`)."""
    links = []
    for icon, label, endpoint, key in NAV_ITEMS:
        cls = "active" if key == active else ""
        links.append(
            f'<a class="{cls}" href="{url_for(endpoint)}">'
            f'<span class="ic">{icon}</span>{label}</a>')
    return ('<nav class="sidebar"><div class="brand">photo-sorter'
            '<small>reconocimiento local</small></div>' + "".join(links) + "</nav>")


def shell(active, inner, head_extra=""):
    """Envuelve el contenido `inner` (HTML+Jinja de la página) con el <head>
    del tema oscuro y la barra lateral. Todo se renderiza junto por Jinja."""
    return (
        '<!doctype html><html lang="es"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        "<title>photo-sorter</title>" + head_extra +
        "<style>" + BASE_CSS + "</style></head><body>"
        '<div class="app">' + _nav_html(active) + '<main class="main">'
        + inner +
        "</main></div></body></html>")


# Metas de auto-refresco (van al <head> vía head_extra) según el estado en curso
REFRESH_STATE = '{% if state.status == "running" %}<meta http-equiv="refresh" content="2">{% endif %}'
REFRESH_IMPORT = '{% if imp.status == "running" %}<meta http-equiv="refresh" content="2">{% endif %}'

HOME_BODY = """
<h1>Ordenar por rostro</h1>
<p class="hint">Al organizar, tus fotos se <b>mueven</b> al destino como <code>nombre_0000.jpg</code>, separadas por persona (desaparecen del origen). Las fotos sin nombre quedan donde estaban. Todo corre en esta máquina.</p>

{% if message %}<div class="msg">{{ message }}</div>{% endif %}
{% if state.status == 'error' %}<div class="err">Error del análisis: {{ state.error }}</div>{% endif %}
{% if state.status == 'done' and groups %}<div class="msg">✅ Análisis terminado: {{ groups|length }} grupo(s) de rostros.</div>{% endif %}
{% if state.status == 'stopped' %}<div class="msg">⏹ Análisis detenido. {% if groups %}Se guardaron {{ groups|length }} grupo(s) de las fotos procesadas hasta ese punto.{% else %}No se llegó a procesar ninguna foto con rostro.{% endif %}</div>{% endif %}

<div class="panel">
  <form method="post" action="{{ url_for('do_analyze') }}">
    <div class="row">
      <label>Origen:</label>
      <input class="path" type="text" name="photos_dir" value="{{ cfg.get('photos_dir','') }}"
             placeholder="Carpeta con las fotos a clasificar">
      <button type="button" onclick="browse('photos_dir')">📂 Elegir carpeta…</button>
    </div>
    <div class="row">
      <span class="meta">📚 Se organizan a la Biblioteca: {% if library %}<code>{{ library }}</code>{% else %}<b>sin elegir</b>{% endif %}</span>
      <a href="{{ url_for('settings_page') }}" class="meta">cambiar en ⚙ Ajustes →</a>
    </div>
    <div class="row">
      <label>Revisar:</label>
      <select name="media">
        <option value="images" {% if cfg.get('media', 'both') == 'images' %}selected{% endif %}>Solo imágenes (rápido)</option>
        <option value="videos" {% if cfg.get('media', 'both') == 'videos' %}selected{% endif %}>Solo videos</option>
        <option value="both" {% if cfg.get('media', 'both') == 'both' %}selected{% endif %}>Imágenes y videos</option>
      </select>
      <span class="meta">los videos son mucho más lentos (1 fotograma por segundo)</span>
    </div>
    <div class="row">
      <label>Tamaño mín. de cara:</label>
      <select name="min_face">
        <option value="48" {% if cfg.get('min_face', 80)|int == 48 %}selected{% endif %}>Chico (48px) — incluye caras lejanas</option>
        <option value="80" {% if cfg.get('min_face', 80)|int == 80 %}selected{% endif %}>Medio (80px) — recomendado</option>
        <option value="120" {% if cfg.get('min_face', 80)|int == 120 %}selected{% endif %}>Grande (120px) — solo primer plano</option>
        <option value="160" {% if cfg.get('min_face', 80)|int == 160 %}selected{% endif %}>Muy grande (160px) — solo retratos</option>
      </select>
      <span class="meta">más grande = menos gente de fondo, muchos menos grupos</span>
    </div>
    <div class="row">
      <button class="primary"
              {% if state.status == 'running' %}disabled{% endif %}
              {% if groups %}onclick="return confirm('Volver a analizar rehace TODO desde cero y borra los nombres y ajustes de este análisis (las personas conocidas se conservan). ¿Seguir?')"{% endif %}>
        🔍 Analizar fotos del origen
      </button>
    </div>
    <span class="meta">Detecta los rostros (incluso de perfil o anguladas) y agrupa las caras iguales. Con muchas fotos tarda.</span>
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

<div class="panel">
  <div class="counter">👤 Personas conocidas: <b>{{ known_count }}</b>
    <a href="{{ url_for('known_page') }}" style="font-size:.9rem;margin-left:.5rem">ir a la galería →</a></div>
  <p class="meta">En cada análisis, el programa sugiere el nombre de estas personas en los grupos que se parecen
     (vos confirmás o corregís). Cada persona que nombres se suma sola. Para importar gente desde una carpeta
     ya ordenada o cambiar portadas, entrá a la <a href="{{ url_for('known_page') }}">galería</a>.</p>
</div>

{% if groups %}
<div class="panel">
  <div class="counter">Se encontraron <b>{{ groups|length }}</b> grupo(s) de rostros —
    <span id="labeled-count">{{ labeled }}</span> con nombre,
    <span id="pending-count"></span> por revisar.</div>
  <div class="meta" id="stats-line" {% if not (stats.cur_total or stats.hist_total) %}style="display:none"{% endif %}>
    🎯 Sugerencias acertadas:
    {%- if stats.cur_total %} {{ stats.cur_hits }}/{{ stats.cur_total }} ({{ (100 * stats.cur_hits / stats.cur_total)|round|int }}%) en este análisis
    {%- else %} todavía ninguna en este análisis{% endif %}
    {%- if stats.hist_total %} · histórico {{ (100 * stats.hist_hits / stats.hist_total)|round|int }}% ({{ stats.hist_total }} grupo(s)){% endif %}
  </div>
  <form method="post" action="{{ url_for('do_organize') }}">
    <button class="organize" id="organize-btn" {% if not library %}disabled{% endif %}
            onclick="return confirm('Se moverán las fotos con nombre a la Biblioteca (desaparecen del origen) y esos grupos saldrán de la lista. ¿Continuar?')">
      📁 Organizar: mover fotos a la Biblioteca
    </button>
    <button type="button" class="secondary" onclick="toggleList()">Ver/editar todos los grupos</button>
    <button class="secondary" formaction="{{ url_for('close_analysis') }}"
            onclick="return confirm('Cerrar el análisis: desaparecen los grupos de esta sesión. Las fotos y las personas conocidas no se tocan. ¿Seguir?')">
      ✖ Cerrar análisis
    </button>
    {% if not library %}<span class="meta">Elegí antes la Biblioteca en ⚙ Ajustes.</span>{% endif %}
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
const KNOWN_THUMB_URL = "{{ url_for('known_thumb', name='NAME') }}";
const PERSON_URL = "{{ url_for('person_page', name='NAME') }}";
const MAX_PREVIEW = 12;

function hideBroken(img) { img.style.display = 'none'; }   // persona sin portada
function openPerson(img) {   // clic en la portada: abre la galería de esa persona
  window.open(PERSON_URL.replace('NAME', encodeURIComponent(img.dataset.person)));
}

let queue = GROUPS.filter(g => !g.name).map(g => g.id);   // grupos sin nombre, en orden
let pos = 0;
let wizardManual = false;   // true = mostrar el campo de texto (elegiste "Otra")
let sugIdx = 0;             // candidato actual dentro de las sugerencias del grupo

function curSuggestion(g) {
  return (g.suggestions && sugIdx < g.suggestions.length) ? g.suggestions[sugIdx] : null;
}

function curConf(g) {   // % de confianza del candidato actual (si el análisis lo trae)
  return (g.sug_conf && sugIdx < g.sug_conf.length) ? g.sug_conf[sugIdx] : null;
}

function refreshStats(s) {   // línea "🎯 Sugerencias acertadas" sin recargar
  const el = document.getElementById('stats-line');
  if (!el || !s || !(s.cur_total || s.hist_total)) return;
  let txt = '🎯 Sugerencias acertadas: ';
  txt += s.cur_total ?
    s.cur_hits + '/' + s.cur_total + ' (' + Math.round(100 * s.cur_hits / s.cur_total) + '%) en este análisis' :
    'todavía ninguna en este análisis';
  if (s.hist_total) txt += ' · histórico ' + Math.round(100 * s.hist_hits / s.hist_total) + '% (' + s.hist_total + ' grupo(s))';
  el.textContent = txt;
  el.style.display = '';
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
      // hay candidato: mostrar su portada y preguntar ¿Es X? con Sí / No / Otra
      '<div class="row">' +
        '<img class="ask-cover" alt="" title="Portada de ' + curSuggestion(g) +
          ' — tocala para ver sus fotos" data-person="' + curSuggestion(g) +
          '" onerror="hideBroken(this)" onclick="openPerson(this)" style="cursor:pointer" src="' +
          KNOWN_THUMB_URL.replace('NAME', encodeURIComponent(curSuggestion(g))) + '?v=' + Date.now() + '">' +
        '<span class="ask">¿Es <b>' + curSuggestion(g) + '</b>?' +
        (curConf(g) !== null ? ' <span class="meta">(' + curConf(g) + '% seguro)</span>' : '') + '</span>' +
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
  }).then(r => r.json()).then(d => { refreshStats(d.stats); return d; });
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
        (g.suggestions && g.suggestions.length && !g.name ? ' · <b>💡 sugerencia: ' + g.suggestions[0] +
          (g.sug_conf && g.sug_conf.length ? ' (' + g.sug_conf[0] + '%)' : '') + '</b>' : '') +
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
"""


DUPES_TEMPLATE = """
<h1>Buscar duplicados</h1>
<p class="hint">Encuentra fotos repetidas: <b>exactas</b> (mismo archivo con otro nombre) y <b>parecidas</b> (la misma imagen reescalada o recomprimida). En cada grupo se marca cuál conviene conservar (mayor resolución). Al resolver, las sobrantes se <b>mueven</b> a una carpeta <code>_duplicados</code> dentro de la carpeta elegida — no se borran, las revisás vos.</p>

{% if message %}<div class="msg">{{ message }}</div>{% endif %}
{% if state.status == 'error' %}<div class="err">Error: {{ state.error }}</div>{% endif %}

<div class="panel">
  <form method="post" action="{{ url_for('dupes_scan') }}">
    <div class="row">
      <label><b>Carpeta:</b></label>
      <input class="path" type="text" name="folder" value="{{ folder }}"
             placeholder="Carpeta a revisar (incluye subcarpetas)">
      <button type="button" onclick="browse('folder')">📂 Elegir carpeta…</button>
    </div>
    <button class="primary" {% if state.status == 'running' %}disabled{% endif %}>
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
  <div class="row">
    <form method="post" action="{{ url_for('dupes_clear') }}" style="margin:0">
      <button class="secondary">🧽 Limpiar resultados</button>
    </form>
    <span class="meta">Vacía esta lista y vuelve a la pantalla en blanco (no toca ninguna foto).</span>
  </div>
  <form method="post" action="{{ url_for('dupes_resolve') }}">
    <div class="panel">
      <p><b>{{ groups|length }} grupo(s)</b> de duplicados — {{ surplus }} foto(s) sobrantes.
         Se movería lo tildado; las marcadas "conservar" quedan.</p>
      <button class="danger" onclick="return confirm('Se moverán las fotos tildadas a la carpeta _duplicados dentro del origen. ¿Continuar?')">
        🧹 Mover {{ surplus }} sobrante(s) a _duplicados
      </button>
    </div>
    {% set ns = namespace(i=0) %}
    {% for g in groups %}
    <div class="group">
      <div class="meta">Grupo de {{ g.files|length }} — {{ 'exactas' if g.exact else 'parecidas' }} · clic en una imagen abre el visor (zoom, arrastre, ← →)</div>
      <div class="cards">
        {% for f in g.files %}
        <div class="thumb {% if f.keeper %}keeper{% endif %}">
          <img src="{{ url_for('dupe_thumb', idx=f.idx) }}" alt="foto" style="cursor:zoom-in"
               onclick="lbOpen({{ ns.i }})" title="{{ f.rel }}">
          <div>{{ f.w }}×{{ f.h }} · {{ (f.size/1024)|round|int }} KB</div>
          {% if f.keeper %}
            <span class="badge b-keep">conservar</span>
          {% else %}
            <span class="badge {{ 'b-exact' if f.exact else 'b-sim' }}">{{ 'exacta' if f.exact else 'parecida' }}</span>
            <div><label><input type="checkbox" name="rel" value="{{ f.rel }}" checked> mover</label></div>
          {% endif %}
        </div>
        {% set ns.i = ns.i + 1 %}
        {% endfor %}
      </div>
    </div>
    {% endfor %}
  </form>
  <script>
  window.LB_PHOTOS = [
    {% for g in groups %}{% for f in g.files %}{url:"{{ url_for('dupe_photo', rel=f.rel) }}",name:"{{ f.rel|e }} ({{ f.w }}×{{ f.h }})"},{% endfor %}{% endfor %}
  ];
  </script>
  __LIGHTBOX__
  {% else %}
    <div class="panel">✅ No se encontraron duplicados.
      <form method="post" action="{{ url_for('dupes_clear') }}" style="display:inline; margin-left:.5rem">
        <button class="secondary">🧽 Limpiar</button>
      </form>
    </div>
  {% endif %}
{% endif %}
<script>
function browse(field) {
  fetch('{{ url_for('browse') }}')
    .then(r => r.json())
    .then(d => { if (d.path) document.getElementsByName(field)[0].value = d.path; });
}
</script>
"""


RENAME_TEMPLATE = """
<h1>Corregir carpeta (renombrar)</h1>
<p class="hint">Renombra en masa las imágenes y videos de una carpeta con el patrón que elijas: una palabra, un separador y un número que se va sumando solo (<code>0000</code>, <code>0001</code>…). Solo cambia nombres (no toca el contenido). Trabaja solo en el primer nivel de la carpeta (no entra en subcarpetas).</p>

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


def _confidence(dist):
    """Traduce la distancia coseno de una sugerencia a un % legible. Curva
    logística centrada en el umbral estricto de reconocimiento (known.MATCH_EPS
    = 0.50 -> 50%); por debajo sube rápido (0.30 -> ~92%), por encima cae. Es
    una calibración para leer la distancia de un vistazo, no una probabilidad
    exacta."""
    conf = 100.0 / (1.0 + math.exp((dist - 0.50) / 0.08))
    return max(1, min(99, int(round(conf))))


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
                "sug_conf": [_confidence(d) for d in c.get("sug_dist", [])],  # % por candidato
                "review": c.get("review", False),  # grupo "Para revisar" (depurado)
                "suspects": len(suspects),
            })

    known = load_known()
    return render_template_string(
        shell("home", HOME_BODY, head_extra=REFRESH_STATE),
        cfg=cfg,
        state=STATE,
        groups=groups,
        labeled=sum(1 for g in groups if g["name"]),
        known_count=len(known),
        library=get_library(cfg),
        message=request.args.get("msg", ""),
        stats=stats_summary(),
    )


def _clear_analysis():
    """Borra el análisis anterior (grupos, nombres, vectores y miniaturas).
    Antes de borrar, los aciertos de sugerencias pasan al histórico de calidad."""
    archive_current()
    CLUSTERS_PATH.unlink(missing_ok=True)
    LABELS_PATH.unlink(missing_ok=True)
    ENCODINGS_NPY.unlink(missing_ok=True)
    if FACES_DIR.is_dir():
        import shutil
        shutil.rmtree(FACES_DIR, ignore_errors=True)


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


def _analysis_worker(photos_dir, exclude, min_face_px, media):
    def progress(current, total, photo):
        STATE.update(current=current, total=total, photo=photo)
    def should_stop():
        return STATE.get("cancel", False)
    try:
        run_analysis(photos_dir, exclude=exclude, progress=progress,
                     should_stop=should_stop, min_face_px=min_face_px, media=media)
        STATE["status"] = "stopped" if STATE.get("cancel") else "done"
    except Exception as e:
        STATE.update(status="error", error=str(e))


@app.route("/analyze", methods=["POST"])
def do_analyze():
    if STATE["status"] == "running":
        return redirect(url_for("home"))

    # el Origen se elige acá mismo (cambia en cada análisis); la Biblioteca es fija
    cfg = load_config()
    origen = request.form.get("photos_dir", "").strip()
    if not origen or not Path(origen).is_dir():
        return redirect(url_for("home", msg="Esa carpeta de Origen no existe."))
    photos_dir = Path(origen)

    # si cambió el origen, el análisis viejo (grupos/nombres) ya no aplica
    if origen != cfg.get("photos_dir", ""):
        _clear_analysis()

    # tamaño mínimo de cara y qué revisar (se recuerdan para la próxima)
    try:
        min_face = max(24, min(400, int(request.form.get("min_face", MIN_FACE_PX))))
    except ValueError:
        min_face = MIN_FACE_PX
    media = request.form.get("media", "both")
    if media not in ("images", "videos", "both"):
        media = "both"
    cfg["photos_dir"] = origen
    cfg["min_face"] = min_face
    cfg["media"] = media
    save_config(cfg)

    # no re-analizar lo que ya está ordenado en la Biblioteca
    library = get_library(cfg)
    exclude = Path(library) if library else None
    STATE.update(status="running", current=0, total=0, photo="", error="", cancel=False)
    threading.Thread(target=_analysis_worker, args=(photos_dir, exclude, min_face, media),
                     daemon=True).start()
    return redirect(url_for("home"))


@app.route("/analyze/stop", methods=["POST"])
def stop_analyze():
    if STATE["status"] == "running":
        STATE["cancel"] = True
    return redirect(url_for("home"))


@app.route("/analysis/close", methods=["POST"])
def close_analysis():
    """Cierra la sesión de análisis: se van los grupos, nombres y miniaturas
    (los aciertos pasan al histórico). No toca fotos ni personas conocidas."""
    if STATE["status"] == "running":
        return redirect(url_for("home"))
    _clear_analysis()
    STATE.update(status="idle", current=0, total=0, photo="", error="")
    return redirect(url_for("home", msg="Análisis cerrado."))


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
    """Aprende las caras de la Biblioteca (una subcarpeta por persona)."""
    if IMPORT_STATE["status"] == "running":
        return redirect(url_for("known_page"))
    folder = get_library()
    if not folder or not Path(folder).is_dir():
        return redirect(url_for("known_page",
                                msg="Elegí primero la Biblioteca (pestaña Ajustes)."))
    IMPORT_STATE.update(status="running", current=0, total=0, photo="", error="")
    threading.Thread(target=_import_worker, args=(folder,), daemon=True).start()
    return redirect(url_for("known_page"))


@app.route("/known/import/dismiss", methods=["POST"])
def known_import_dismiss():
    """Cierra el cartel del resultado de la importación (✕ o timer de la web)."""
    if IMPORT_STATE["status"] != "running":
        IMPORT_STATE.update(status="idle", error="")
        IMPORT_STATE.pop("added", None)
    return {"ok": True}


@app.route("/known/clear", methods=["POST"])
def known_clear():
    KNOWN_PATH.unlink(missing_ok=True)
    if KNOWN_THUMBS.is_dir():
        import shutil
        shutil.rmtree(KNOWN_THUMBS, ignore_errors=True)
    return redirect(url_for("known_page", msg="Base de personas conocidas vaciada."))


KNOWN_TEMPLATE = """
<h1>Galería</h1>
<p class="hint">Estas son las personas que el programa ya reconoce solo. Tocá una persona para ver sus fotos y elegir cuál se muestra como portada.</p>
{% if message %}<div class="msg">{{ message }}</div>{% endif %}

{% if not known_dir %}
<div class="err">Todavía no elegiste la <b>Biblioteca</b>. Andá a
  <a href="{{ url_for('settings_page') }}">⚙ Ajustes</a> y elegí la carpeta donde viven tus
  personas (una subcarpeta por cada una).</div>
{% else %}
<div class="panel">
  <div class="row">
    <span class="meta">📚 Biblioteca: <code>{{ known_dir }}</code></span>
    <a href="{{ url_for('settings_page') }}" class="meta">cambiar en Ajustes →</a>
  </div>
  <form method="post" action="{{ url_for('known_import') }}">
    <div class="row">
      <button class="primary" {% if imp.status == 'running' %}disabled{% endif %}>
        📥 Aprender las caras de la Biblioteca
      </button>
      {% if people %}
      <button formaction="{{ url_for('known_clear') }}"
              onclick="return confirm('¿Vaciar la base de personas conocidas? (no toca las fotos)')">Vaciar base</button>
      {% endif %}
    </div>
    <span class="meta">Aprende de cada subcarpeta y suma personas nuevas sin reetiquetar. Se puede
      repetir: a los ya conocidos solo les suma ejemplos. Saltea <code>_grupo</code> y <code>_revisar</code>,
      así solo aprende de fotos donde la persona está sola.</span>
  </form>
</div>
{% endif %}

{% if imp.status == 'running' %}
<div class="panel">
  <p>Aprendiendo {{ imp.current }}/{{ imp.total }}: {{ imp.photo }}</p>
  <progress value="{{ imp.current }}" max="{{ imp.total or 1 }}"></progress>
  <p class="meta">Esta página se actualiza sola cada 2 segundos.</p>
</div>
{% elif imp.status == 'error' %}
<div class="err"><button type="button" class="dismiss" onclick="dismissImport(this)" title="Cerrar">✕</button>
  Error al importar: {{ imp.error }}</div>
{% elif imp.status == 'done' and imp.get('added') %}
<div class="msg" id="import-msg"><button type="button" class="dismiss" onclick="dismissImport(this)" title="Cerrar">✕</button>
  Importadas {{ imp.added|length }} persona(s):
  {{ imp.added.keys()|list|join(', ') }}.</div>
<script>/* el cartel se cierra solo a los 15 segundos (o con la ✕) */
  setTimeout(() => { const b = document.querySelector('#import-msg .dismiss'); if (b) dismissImport(b); }, 15000);
</script>
{% endif %}

<script>
function browse(field) {
  fetch('{{ url_for('browse') }}')
    .then(r => r.json())
    .then(d => { if (d.path) document.getElementsByName(field)[0].value = d.path; });
}

function dismissImport(btn) {   // cierra el cartel de importación (✕ o timer)
  fetch('{{ url_for('known_import_dismiss') }}', {method: 'POST'});
  const box = btn.closest('.msg, .err');
  if (box) box.remove();
}

function norm(s) {   // minúsculas y sin tildes, para buscar sin exactitud
  return s.toLowerCase().normalize('NFD').replace(/[\\u0300-\\u036f]/g, '');
}
const ALL_TAGS = {{ all_tags|tojson }};
function filterPeople() {
  const raw = document.getElementById('pfilter').value.trim();
  const tokens = raw.split(',').map(t => norm(t.trim())).filter(Boolean);
  // qué tokens son etiquetas reales (para arrastrar el filtro al perfil)
  const tagToks = tokens.filter(t => ALL_TAGS.some(x => norm(x).includes(t)));
  const tagsParam = tagToks.length ? '?tags=' + encodeURIComponent(tagToks.join(',')) : '';
  let visibles = 0;
  document.querySelectorAll('.person[data-name]').forEach(card => {
    const hay = norm(card.dataset.name + ' ' + (card.dataset.tags || ''));
    const ok = tokens.every(t => hay.includes(t));
    card.style.display = ok ? '' : 'none';
    if (ok) {
      visibles++;
      // al entrar, arrastrar las etiquetas buscadas para pre-filtrar sus fotos
      const a = card.querySelector('a');
      if (a) a.href = card.dataset.href + tagsParam;
    }
  });
  const c = document.getElementById('pcount');
  if (c) c.textContent = visibles;
}
</script>

{% if people %}
<div class="row" style="margin-bottom:.75rem">
  <input type="text" id="pfilter" class="path" placeholder="🔍 Buscar por nombre o etiqueta (coma para varias)…"
         oninput="filterPeople()" autofocus>
  <span class="meta"><span id="pcount">{{ people|length }}</span> persona(s)</span>
</div>
<datalist id="tagsug">{% for t in all_tags %}<option value="{{ t }}">{% endfor %}</datalist>
<div class="grid">
  {% for p in people %}
  <div class="person" id="p{{ loop.index0 }}" data-name="{{ p.name }}"
       data-tags="{{ p.tags|join(' ') }}" data-href="{{ url_for('person_page', name=p.name) }}">
    <a href="{{ url_for('person_page', name=p.name) }}" title="Ver las fotos de {{ p.name }}">
    {% if p.thumb %}
      <img src="{{ url_for('known_thumb', name=p.name) }}" alt="{{ p.name }}">
    {% else %}
      <div class="noimg">👤</div>
    {% endif %}
    <div class="pname">{{ p.name }}</div>
    </a>
    {% if p.tags %}<div class="pcount">🏷 {{ p.tags|join(', ') }}</div>{% endif %}
    <div class="pcount">{{ p.count }} cara(s) aprendida(s)</div>
    <div class="pcount">{% if p.corrected %}🧹 <b style="color:var(--grn)">corregido</b> {{ p.corrected }}{% else %}<span style="opacity:.6">🧹 sin corregir</span>{% endif %}</div>
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

{% if notfaces %}
<p style="margin-top:1.5rem">🚫 <b>{{ notfaces }}</b> detección(es) marcadas como "no es una persona"
  (tapizados, objetos): en los próximos análisis se descartan solas.
  <form method="post" action="{{ url_for('notfaces_clear') }}" style="display:inline"
        onsubmit="return confirm('¿Vaciar la lista negra? Lo descartado puede volver a aparecer en próximos análisis.')">
    <button>Vaciar lista negra</button>
  </form>
</p>
{% endif %}
"""


@app.route("/known")
def known_page():
    people = known_summary()
    tagsets = tags_mod.people_tagsets()
    for p in people:   # "corregido hace X" + etiquetas (suyas + de sus fotos)
        p["corrected"] = person_meta.humanize(person_meta.get(p["name"])["corrected"])
        p["tags"] = tagsets.get(p["name"], [])
    return render_template_string(
        shell("known", KNOWN_TEMPLATE, head_extra=REFRESH_IMPORT),
        people=people, notfaces=count_not_faces(),
        known_dir=get_library(), all_tags=tags_mod.all_tags(),
        imp=IMPORT_STATE,
        message=request.args.get("msg", ""))


SETTINGS_TEMPLATE = """
<h1>Ajustes</h1>
<p class="hint">La <b>Biblioteca</b>: la carpeta fija donde viven tus personas ordenadas
   (<code>Biblioteca/&lt;persona&gt;/</code>). El <b>Origen</b> del análisis se elige en cada corrida
   desde <a href="{{ url_for('home') }}">Ordenar por rostro</a>, porque cambia seguido.</p>

{% if message %}<div class="msg">{{ message }}</div>{% endif %}

<div class="panel">
  <form method="post" action="{{ url_for('settings_save') }}">
    <div class="row">
      <label>📚 Biblioteca:</label>
      <input class="path" type="text" name="library" value="{{ library }}"
             placeholder="Carpeta con una subcarpeta por persona">
      <button type="button" onclick="browse('library')">📂 Elegir carpeta…</button>
    </div>
    <p class="meta">Es <b>una sola</b> y la usan todos: el <b>organizador</b> y el <b>buscador</b> mueven
      acá; la <b>galería</b>, el <b>import</b>, el <b>corrector</b> y el <b>separador</b> leen de acá.</p>

    <button class="primary">💾 Guardar</button>
  </form>
</div>

<div class="panel">
  <div class="counter">💬 Comentarios de las fotos</div>
  <p class="meta">Cuando abrís una foto de la galería, la IA le puede escribir comentarios
     como si fueran de seguidores. Acá elegís con qué onda los escribe y con qué nombres
     los firma.</p>
  <form method="post" action="{{ url_for('settings_save') }}">
    <input type="hidden" name="library" value="{{ library }}">
    <div class="row">
      <label>Tono:</label>
      <select name="comment_tone">
        {% for k, v in tones.items() %}
        <option value="{{ k }}" {% if k == comment_tone %}selected{% endif %}>{{ k }} — {{ v }}</option>
        {% endfor %}
      </select>
    </div>
    <div class="row">
      <label>@usuarios:</label>
      <input class="path" type="text" name="comment_users" value="{{ comment_users }}"
             placeholder="separados por coma">
    </div>
    <p class="meta">Los nombres los pongo yo de esta lista, no los inventa la IA: un modelo
       chico devuelve "@usuario1", "@usuario2" y queda feo. Vaciá el campo para volver a
       los de fábrica.</p>
    <button class="primary">💾 Guardar</button>
  </form>
</div>

<div class="panel">
  <div class="counter">Estado</div>
  <p class="meta">
    📚 Biblioteca: {% if library %}<code>{{ library }}</code> — <b>{{ n_people }}</b> carpeta(s) de persona{% else %}<b>sin elegir</b>{% endif %}<br>
    📥 Último Origen: {% if cfg.get('photos_dir') %}<code>{{ cfg.get('photos_dir') }}</code>{% else %}<b>ninguno todavía</b>{% endif %}
  </p>
</div>

<script>
function browse(field) {
  fetch('{{ url_for('browse') }}')
    .then(r => r.json())
    .then(d => { if (d.path) document.getElementsByName(field)[0].value = d.path; });
}
</script>
"""


@app.route("/settings")
def settings_page():
    cfg = load_config()
    library = get_library(cfg)
    n_people = 0
    if library and Path(library).is_dir():
        n_people = sum(1 for d in Path(library).iterdir() if d.is_dir())
    return render_template_string(
        shell("settings", SETTINGS_TEMPLATE),
        cfg=cfg, library=library, n_people=n_people,
        tones=posts_mod.TONES,
        comment_tone=cfg.get("comment_tone", posts_mod.DEFAULT_TONE),
        comment_users=", ".join(cfg.get("comment_users") or posts_mod.DEFAULT_USERS),
        message=request.args.get("msg", ""))


@app.route("/settings/save", methods=["POST"])
def settings_save():
    """Guarda la Biblioteca (una sola para todo). El Origen se elige por
    análisis en la página principal."""
    cfg = load_config()
    library = request.form.get("library", "").strip()
    if library and not Path(library).is_dir():
        return redirect(url_for("settings_page",
                                msg=f"Esa carpeta de Biblioteca no existe: {library}"))
    set_library(cfg, library)

    # comentarios: los dos formularios de Ajustes postean acá, así que cada campo
    # se toca solo si vino (si no, guardar uno pisaría lo del otro)
    if "comment_tone" in request.form:
        tone = request.form.get("comment_tone", "")
        if tone in posts_mod.TONES:
            cfg["comment_tone"] = tone
    if "comment_users" in request.form:
        users = [u.strip().lstrip("@") for u in request.form["comment_users"].split(",")]
        users = [u for u in users if u]
        cfg["comment_users"] = users        # vacío = los de fábrica

    save_config(cfg)
    return redirect(url_for("settings_page", msg="Ajustes guardados."))


@app.route("/notfaces/clear", methods=["POST"])
def notfaces_clear():
    clear_not_faces()
    return redirect(url_for("known_page", msg="Lista negra vaciada."))


LINKS_TEMPLATE = """
<h1>Extraer enlaces</h1>
<p class="hint">Pegá la dirección de una página y el texto a buscar: la página se abre en un navegador invisible (funciona también con páginas que cargan por JavaScript) y se listan todos los enlaces de su código que contengan ese texto. Como cada página es distinta, el texto lo elegís vos en cada búsqueda.</p>

{% if error %}<div class="err">{{ error }}</div>{% endif %}

<div class="panel">
  <form method="post" onsubmit="const b=document.getElementById('go'); b.disabled=true; b.textContent='Cargando la página…'">
    <div class="row">
      <label>Página:</label>
      <input class="path" type="text" name="page_url" value="{{ page_url }}" placeholder="https://ejemplo.com/galeria">
    </div>
    <div class="row">
      <label>Buscar:</label>
      <input type="text" name="term" value="{{ term }}" placeholder="ej: mp4, /video/, download">
      <button class="primary" id="go">🔗 Buscar enlaces</button>
    </div>
    <span class="meta">La página tarda unos segundos en cargarse (se hace scroll para destrabar galerías con carga diferida).</span>
  </form>
</div>

{% if links is not none %}
<div class="panel">
  <div class="counter">{{ links|length }} enlace(s) con "{{ term }}"</div>
  {% if links %}
  <div class="row">
    <button class="primary" onclick="copyAll()">📋 Copiar todos (separados por coma)</button>
    <span class="saved-tick" id="copied">✓ copiados</span>
  </div>
  <ol>
    {% for u in links %}<li><a href="{{ u }}" target="_blank" rel="noreferrer">{{ u }}</a></li>{% endfor %}
  </ol>
  {% else %}
  <p class="meta">No apareció ese texto en ningún enlace. Puede que la página pida iniciar sesión,
     que el contenido tarde más en cargar, o que el texto esté escrito distinto (probá con un pedazo más corto).</p>
  {% endif %}
</div>
<script>
const LINKS = {{ links|tojson }};
function copyAll() {
  navigator.clipboard.writeText(LINKS.join(', ')).then(() => {
    const t = document.getElementById('copied');
    t.classList.add('show');
    setTimeout(() => t.classList.remove('show'), 1500);
  });
}
</script>
{% endif %}
"""


FINDER_TEMPLATE = """
<h1>Buscar persona</h1>
<p class="hint">Elegí una persona conocida y una carpeta: se revisan todas sus fotos y videos buscando esa cara.
   Después podés mover los encontrados a la carpeta de la persona (la de las galerías).</p>

{% if message %}<div class="msg">{{ message }}</div>{% endif %}
{% if state.status == 'error' %}<div class="err">{{ state.error }}</div>{% endif %}

<div class="panel">
  <form method="post" action="{{ url_for('finder_run') }}">
    <div class="row">
      <label>Persona:</label>
      <select name="person" {% if not people %}disabled{% endif %}>
        <option value="*" {% if sel == '*' %}selected{% endif %}>👥 Todos los conocidos</option>
        {% for p in people %}<option value="{{ p }}" {% if p == sel %}selected{% endif %}>{{ p }}</option>{% endfor %}
      </select>
      {% if not people %}<span class="meta">Todavía no hay personas conocidas (importá o nombrá gente primero).</span>{% endif %}
    </div>
    <div class="row">
      <label>Carpeta:</label>
      <input class="path" type="text" name="folder" value="{{ state.folder }}"
             placeholder="Carpeta a revisar (incluye subcarpetas)">
      <button type="button" onclick="browse('folder')">📂 Elegir carpeta…</button>
    </div>
    <div class="row">
      <label>Rigor:</label>
      <span class="meta">estricto</span>
      <input type="range" name="eps" min="{{ (eps_min*100)|int }}" max="{{ (eps_max*100)|int }}"
             value="{{ (state.eps*100)|int }}"
             oninput="document.getElementById('epsv').textContent = (this.value/100).toFixed(2)">
      <span class="meta">permisivo</span>
      <b id="epsv">{{ '%.2f'|format(state.eps) }}</b>
      <span class="meta">(0.50 recomendado; subilo si le faltan fotos, bajalo si le sobran)</span>
    </div>
    <div class="row">
      <label>Revisar:</label>
      <select name="media">
        <option value="images" {% if state.media == 'images' %}selected{% endif %}>Solo imágenes (rápido)</option>
        <option value="videos" {% if state.media == 'videos' %}selected{% endif %}>Solo videos</option>
        <option value="both" {% if state.media == 'both' %}selected{% endif %}>Imágenes y videos</option>
      </select>
      <span class="meta">los videos son mucho más lentos (1 fotograma por segundo)</span>
    </div>
    <button class="primary" {% if state.status == 'running' or not people %}disabled{% endif %}>
      🔍 Buscar a esa persona en la carpeta
    </button>
  </form>
  {% if state.status == 'running' %}
    <p>Revisando <b>{{ state.current }}/{{ state.total }}</b>: {{ state.photo }}</p>
    <progress value="{{ state.current }}" max="{{ state.total or 1 }}"></progress>
    <form method="post" action="{{ url_for('finder_stop') }}" style="margin-top:.5rem">
      <button class="danger">⛔ Detener búsqueda</button>
    </form>
    <p class="meta">Esta página se actualiza sola cada 2 segundos.</p>
  {% endif %}
</div>

{% if state.status == 'done' and state.mode == 'one' %}
<div class="panel">
  <div class="counter">{{ state.results|length }} de {{ state.scanned }} archivo(s) tienen a <b>{{ state.name }}</b></div>
  {% if state.results %}
  <form method="post" action="{{ url_for('finder_move') }}"
        onsubmit="return confirm('Se moverán los tildados a la carpeta de {{ state.name }}. Los que destildaste se descartan de esta lista. ¿Continuar?')">
    <input type="hidden" name="person" value="{{ state.name }}">
    <div class="row">
      <button type="button" class="secondary" onclick="toggleAll(this)">Marcar / desmarcar todos</button>
      {% if not known_dir %}<span class="meta">Configurá antes la Biblioteca (⚙ Ajustes).</span>
      {% else %}<span class="meta">Destino: {{ known_dir }}\\{{ state.name }} · destildá las que no son y tocá Mover abajo</span>{% endif %}
    </div>
    <div class="grid">
      {% for r in state.results %}
      <label class="hit">
        <input type="checkbox" name="rel" value="{{ r.rel }}" checked>
        <span class="cf">{{ conf(r.dist) }}%</span><br>
        <img src="{{ url_for('find_thumb', fn=r.thumb) }}" alt="">
        <div class="fn" title="{{ r.rel }}">
          <a href="{{ url_for('find_file', rel=r.rel) }}" target="_blank">{{ r.rel.split('/')[-1] }}</a>
        </div>
      </label>
      {% endfor %}
    </div>
    <div class="row" style="margin-top:.75rem">
      <button class="organize" {% if not known_dir %}disabled{% endif %}>
        📦 Mover los tildados a la carpeta de {{ state.name }}
      </button>
      <span class="meta">al mover, esta lista se cierra (las que descartaste no vuelven)</span>
    </div>
  </form>
  {% elif state.cleared %}
  <p class="meta">✅ Listo. Buscá otra persona o cambiá la carpeta arriba.</p>
  {% else %}
  <p class="meta">No se encontró a {{ state.name }} en esa carpeta con el rigor actual.
     Probá subir el deslizador hacia "permisivo" y buscar de nuevo.</p>
  {% endif %}
</div>
{% elif state.status == 'done' %}
<div class="panel">
  <div class="counter">{{ state.by_person|length }} conocido(s) encontrados en {{ state.scanned }} archivo(s) revisados</div>
  {% if not state.by_person %}
  <p class="meta">Ningún conocido apareció en esa carpeta con el rigor actual.
     Probá subir el deslizador hacia "permisivo" y buscar de nuevo.</p>
  {% else %}
  <p class="meta">Un archivo donde aparecen dos conocidos figura en los dos grupos: al moverlo
     se copia a la carpeta de cada uno y el original desaparece recién con el último.</p>
  <div class="row" style="margin-top:.5rem">
    <button type="button" class="secondary" onclick="foldAll(false)">⊟ Colapsar todos</button>
    <button type="button" class="secondary" onclick="foldAll(true)">⊞ Expandir todos</button>
    <span class="meta">tocá el nombre de una persona para plegarla y sacarla del scroll</span>
  </div>
  {% endif %}
</div>
{% for grp in state.by_person %}
<div class="panel">
  <form method="post" action="{{ url_for('finder_move') }}"
        onsubmit="return confirm('Se moverán los tildados a la persona elegida. Los que destildaste se descartan de esta lista. ¿Continuar?')">
    <input type="hidden" name="group" value="{{ grp.name }}">
    <details class="grp" open>
      <summary class="counter">{{ grp.name }} — {{ grp.hits|length }} archivo(s)</summary>
      <div class="row">
        <button type="button" class="secondary" onclick="toggleAll(this)">Marcar / desmarcar todos</button>
        {% if not known_dir %}<span class="meta">Configurá antes la Biblioteca (⚙ Ajustes).</span>
        {% else %}<span class="meta">destildá las que no son, y si detectó mal la persona, cambiala abajo</span>{% endif %}
      </div>
      <div class="grid">
        {% for r in grp.hits %}
        <label class="hit">
          <input type="checkbox" name="rel" value="{{ r.rel }}" checked>
          <span class="cf">{{ conf(r.dist) }}%</span><br>
          <img src="{{ url_for('find_thumb', fn=r.thumb) }}" alt="">
          <div class="fn" title="{{ r.rel }}">
            <a href="{{ url_for('find_file', rel=r.rel) }}" target="_blank">{{ r.rel.split('/')[-1] }}</a>
          </div>
        </label>
        {% endfor %}
      </div>
      <div class="row" style="margin-top:.75rem">
        <button class="organize" {% if not known_dir %}disabled{% endif %}>📦 Mover los tildados a</button>
        <select name="person">
          <option value="{{ grp.name }}" selected>{{ grp.name }} (lo que detectó)</option>
          {% for p in people %}{% if p != grp.name %}<option value="{{ p }}">{{ p }}</option>{% endif %}{% endfor %}
        </select>
        <span class="meta">al mover, este grupo se cierra (las que descartaste no vuelven)</span>
      </div>
    </details>
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
function toggleAll(btn) {   // marca/desmarca solo las casillas del grupo del botón
  const boxes = [...btn.closest('form').querySelectorAll('.grid input[type=checkbox]')];
  const all = boxes.every(b => b.checked);
  boxes.forEach(b => b.checked = !all);
}
function foldAll(open) {   // pliega/despliega todos los acordeones de persona
  document.querySelectorAll('details.grp').forEach(d => d.open = open);
}
</script>
"""


def _finder_worker(folder, name, eps, media):
    def progress(current, total, photo):
        FIND_STATE.update(current=current, total=total, photo=photo)
    def should_stop():
        return FIND_STATE.get("cancel", False)
    try:
        if name == "*":   # todos los conocidos en una sola pasada
            by_person, scanned = find_known(folder, eps, media=media,
                                            progress=progress, should_stop=should_stop)
            FIND_STATE.update(status="done", by_person=by_person, scanned=scanned)
            person_meta.stamp_many(list(load_known().keys()), "searched")
        else:
            results, scanned = find_person(folder, name, eps, media=media,
                                           progress=progress, should_stop=should_stop)
            FIND_STATE.update(status="done", results=results, scanned=scanned)
            person_meta.stamp(name, "searched")
    except Exception as e:
        FIND_STATE.update(status="error", error=str(e))


@app.route("/finder")
def finder_page():
    people = sorted(load_known(), key=str.lower)
    sel = request.args.get("name", "") or FIND_STATE["name"]
    if sel != "*" and sel not in people:
        sel = "*"   # por defecto: todos los conocidos
    return render_template_string(
        shell("finder", FINDER_TEMPLATE, head_extra=REFRESH_STATE),
        people=people,
        sel=sel,
        state=FIND_STATE,
        eps_min=EPS_MIN, eps_max=EPS_MAX,
        known_dir=get_library(),
        conf=_confidence,
        message=request.args.get("msg", ""),
    )


@app.route("/finder/run", methods=["POST"])
def finder_run():
    if FIND_STATE["status"] == "running":
        return redirect(url_for("finder_page"))
    person = request.form.get("person", "").strip()
    folder = request.form.get("folder", "").strip()
    try:
        eps = min(EPS_MAX, max(EPS_MIN, int(request.form.get("eps", "50")) / 100))
    except ValueError:
        eps = EPS_DEFAULT
    media = request.form.get("media", "images")
    if media not in ("images", "videos", "both"):
        media = "images"
    if not person:
        return redirect(url_for("finder_page", msg="Elegí una persona."))
    if not folder or not Path(folder).is_dir():
        return redirect(url_for("finder_page", msg="Esa carpeta no existe."))
    FIND_STATE.update(status="running", current=0, total=0, photo="", error="",
                      results=[], by_person=[], mode="all" if person == "*" else "one",
                      name=person, folder=folder, eps=eps, scanned=0, cancel=False,
                      media=media, cleared=False)
    threading.Thread(target=_finder_worker, args=(folder, person, eps, media), daemon=True).start()
    return redirect(url_for("finder_page"))


@app.route("/finder/stop", methods=["POST"])
def finder_stop():
    if FIND_STATE["status"] == "running":
        FIND_STATE["cancel"] = True
    return redirect(url_for("finder_page"))


@app.route("/find_thumb/<fn>")
def find_thumb(fn):
    return send_from_directory(FIND_THUMBS, fn)


@app.route("/find_file/<path:rel>")
def find_file(rel):
    return send_from_directory(FIND_STATE["folder"], rel)


@app.route("/finder/move", methods=["POST"])
def finder_move():
    """Mueve los archivos elegidos a la carpeta de la persona encontrada."""
    if FIND_STATE["status"] != "done":
        return redirect(url_for("finder_page"))
    known_dir = get_library()
    if not known_dir:
        return redirect(url_for("finder_page", msg="Configurá antes la Biblioteca (⚙ Ajustes)."))
    # person = a quién van (puede ser otra si detectó mal); group = el grupo
    # sobre el que se está trabajando (el que se cierra al terminar)
    person = request.form.get("person", "") or FIND_STATE["name"]
    group = request.form.get("group", person)
    if person not in load_known():
        return redirect(url_for("finder_page", msg="Esa persona no existe en la base."))
    # sin tildados = todas eran falsas: se cierra la sección igual (0 movidas)
    rels = request.form.getlist("rel")

    # Archivos compartidos con OTRO grupo que sigue pendiente: se copian (el
    # original queda hasta que lo mueva el último, como en el organizador).
    copy_only = set()
    if FIND_STATE["mode"] == "all":
        copy_only = {h["rel"] for g in FIND_STATE["by_person"] if g["name"] != group
                     for h in g["hits"]} & set(rels)

    # hits del grupo trabajado, para sacar el vector de cada cara movida
    if FIND_STATE["mode"] == "all":
        hits = next((g["hits"] for g in FIND_STATE["by_person"] if g["name"] == group), [])
    else:
        hits = FIND_STATE["results"]

    moved, errors = move_to_person(FIND_STATE["folder"], known_dir, person, rels, copy_only)
    done = set(moved)

    # ALIMENTA LA BASE: cada cara confirmada por vos (la movés a esta persona)
    # entra como ejemplo. Es dato verificado a mano -> el buscador deja de
    # errar en esas caras. Si corregiste el destino, aprende la persona correcta.
    encs = [h["enc"] for h in hits if h["rel"] in done and h.get("enc")]
    if encs:
        db = load_known()
        add_faces(db, person, [np.asarray(e, dtype=np.float32) for e in encs])
        save_known(db)

    # El grupo trabajado se cierra ENTERO: las tildadas se movieron (a `person`,
    # que puede ser distinta si corregiste), y las destildadas son descartes.
    descartadas = 0
    if FIND_STATE["mode"] == "all":
        for g in FIND_STATE["by_person"]:
            if g["name"] == group:
                descartadas = len(g["hits"]) - len([h for h in g["hits"] if h["rel"] in done])
        FIND_STATE["by_person"] = [g for g in FIND_STATE["by_person"] if g["name"] != group]
    else:
        descartadas = len(FIND_STATE["results"]) - len(done)
        FIND_STATE["results"] = []
        FIND_STATE["cleared"] = True

    corregido = " (corregida)" if group and person != group else ""
    msg = f"{len(moved)} archivo(s) a la carpeta de {person}{corregido}."
    if encs:
        msg += f" {len(encs)} cara(s) sumadas a su base (mejora el reconocimiento)."
    if descartadas:
        msg += f" {descartadas} descartada(s)."
    if copy_only & done:
        msg += f" {len(copy_only & done)} compartidos con otro conocido se copiaron (el original queda hasta mover al último)."
    if errors:
        msg += f" {len(errors)} no se pudieron mover (ej: {errors[0][0]})."
    return redirect(url_for("finder_page", msg=msg))


@app.route("/links", methods=["GET", "POST"])
def links_page():
    """Extractor de enlaces: URL + texto a buscar -> lista de enlaces."""
    page_url = request.form.get("page_url", "").strip()
    term = request.form.get("term", "").strip()
    links = None
    error = ""
    if request.method == "POST":
        if not page_url or not term:
            error = "Completá la dirección de la página y el texto a buscar."
        else:
            if not page_url.lower().startswith(("http://", "https://")):
                page_url = "https://" + page_url
            try:
                links = grab_links(page_url, term)
            except RuntimeError as e:
                error = str(e)
    return render_template_string(
        shell("links", LINKS_TEMPLATE), page_url=page_url, term=term, links=links, error=error)


@app.route("/known_thumb/<name>")
def known_thumb(name):
    return send_from_directory(KNOWN_THUMBS, f"{safe_name(name)}.jpg")


@app.route("/known/forget", methods=["POST"])
def known_forget():
    name = request.form.get("name", "")
    if name:
        forget_person(name)
        # todo lo suyo se va con ella: si no, queda basura colgada de un nombre
        # que ya no existe y reaparece si algún día volvés a crear esa persona
        tags_mod.forget_person(name)      # etiquetas
        posts_mod.forget_person(name)     # pies y comentarios
        person_meta.forget(name)          # fechas y su voz
    return redirect(url_for("known_page", msg=f"Se olvidó a {name}."))


PERSON_TEMPLATE = """
<h1>{{ name }}</h1>
<div class="row" style="justify-content:space-between">
  <a href="{{ url_for('known_page') }}#p{{ anchor }}">← Galería{% if total %} <span class="meta">({{ pos }}/{{ total }})</span>{% endif %}</a>
  <span class="row" style="margin:0">
    {% if prev_name %}<a class="secondary" style="padding:.4rem .8rem;border-radius:6px" href="{{ url_for('person_page', name=prev_name) }}" title="{{ prev_name }}">‹ Anterior</a>
    {% else %}<span class="secondary" style="padding:.4rem .8rem;border-radius:6px;opacity:.45">‹ Anterior</span>{% endif %}
    {% if next_name %}<a class="secondary" style="padding:.4rem .8rem;border-radius:6px" href="{{ url_for('person_page', name=next_name) }}" title="{{ next_name }}">Siguiente ›</a>
    {% else %}<span class="secondary" style="padding:.4rem .8rem;border-radius:6px;opacity:.45">Siguiente ›</span>{% endif %}
  </span>
</div>
{% if message %}<div class="msg">{{ message }}</div>{% endif %}

{% if pick_rel %}
<div class="panel">
  <b>Esa foto tiene {{ pick_count }} caras — ¿cuál es {{ name }}?</b>
  <p class="meta">Tocá la cara correcta: queda de portada y además se suma como
     ejemplo confirmado de {{ name }} (mejora el buscador y las sugerencias).</p>
  <div class="grid">
    {% for i in range(pick_count) %}
    <form method="post" action="{{ url_for('person_cover', name=name) }}" style="margin:0">
      <input type="hidden" name="rel" value="{{ pick_rel }}">
      <input type="hidden" name="face" value="{{ i }}">
      <button style="background:none;border:0;padding:0;cursor:pointer" title="Es esta">
        <img src="{{ url_for('pick_thumb', fn=i|string + '.jpg') }}?v={{ pick_v }}"
             style="height:130px;border-radius:6px" alt="cara {{ i }}">
      </button>
    </form>
    {% endfor %}
  </div>
  <p><a href="{{ url_for('person_page', name=name) }}">Cancelar</a></p>
</div>
{% endif %}

<div class="panel">
  <div class="cover-box">
    {% if has_thumb %}
      <img src="{{ url_for('known_thumb', name=name) }}?v={{ cache_bust }}" alt="portada de {{ name }}">
    {% else %}
      <div class="noimg">👤</div>
    {% endif %}
    <div>
      <b>Portada actual</b>
      <p class="meta">Es la imagen que se muestra en la galería de conocidos y junto al
         "¿Es {{ name }}?" del asistente. Elegí abajo la foto que quieras de portada.</p>
      <p class="meta">
        🧹 Última corrección: <b>{{ corrected_ago or 'nunca' }}</b> ·
        🔍 Última búsqueda: <b>{{ searched_ago or 'nunca' }}</b>
      </p>
      <div style="margin:.4rem 0">
        <span class="meta">🏷 Etiquetas de {{ name }} (valen para todas sus fotos):</span>
        <div class="chips" id="ptags" style="margin-top:.3rem">
          <input class="taginput" id="ptag-input" list="tagsug" placeholder="escribí y Enter…"
                 onkeydown="if(event.key==='Enter'){event.preventDefault();addPersonTag(this.value);this.value=''}">
        </div>
      </div>
      <div style="margin:.6rem 0">
        <span class="meta">🗣 Cómo habla {{ name }}
          <span title="El pie de foto lo escribe ella en primera persona. Esto es lo que la hace sonar distinta de las demás.">ⓘ</span>
        </span>
        <div class="row" style="margin-top:.3rem">
          <input class="path" id="pvoice" value="{{ voice }}"
                 placeholder="ej: sos directa y vulgar, decís las cosas por su nombre…"
                 onkeydown="if(event.key==='Enter'){event.preventDefault();saveVoice()}">
          <button class="secondary" onclick="saveVoice()">💾</button>
          <span class="meta" id="pvoice-st"></span>
        </div>
        <p class="meta">Se le suma al prompt del pie. Vacío = la voz genérica.</p>
      </div>
      <p><a href="{{ url_for('finder_page', name=name) }}">🔍 Buscar sus fotos en otra carpeta →</a></p>
      {% if photos %}
      <form method="post" action="{{ url_for('correct_start', name=name) }}">
        <button title="Revisa la carpeta, confirma las fotos obvias solo y te pregunta por las dudosas. Al final {{ name }} queda solo con caras verificadas.">
          🧹 Corregir entrenamiento (quién es quién en estas fotos)
        </button>
      </form>
      {% endif %}
    </div>
  </div>
</div>

{% if photos or nsfw_count %}
<form class="row" method="get" action="{{ url_for('person_page', name=name) }}" id="ffiltro">
  {% if show_nsfw %}<input type="hidden" name="nsfw" value="1">{% endif %}
  <input type="text" id="ffilter" name="tags" class="path" value="{{ filter_tags }}"
         placeholder="🏷 filtrar sus fotos por etiqueta (coma para varias) + Enter…"
         list="tagsug">
  <button class="secondary">Filtrar</button>
  {% if filter_tags %}
  <a class="secondary" style="padding:.35rem .7rem;border-radius:6px"
     href="{{ url_for('person_page', name=name, nsfw='1' if show_nsfw else None) }}">✕ limpiar</a>
  {% endif %}
  <label class="meta">Por página:</label>
  <select name="per" onchange="this.form.submit()">
    {% for p in per_options %}<option value="{{ p }}" {% if p == per %}selected{% endif %}>{{ p }}</option>{% endfor %}
  </select>
  <span class="meta">
    {% if n_filtradas %}{{ desde }}–{{ hasta }} de <b>{{ n_filtradas }}</b>{% else %}0{% endif %}
    {% if filter_tags %} (filtradas de {{ n_todas }}){% endif %} foto(s)
  </span>
</form>
{# el paginador va arriba y abajo de la grilla; conserva filtro, NSFW y por-página #}
{% macro pager() %}
{% if n_pages > 1 %}
<div class="row" style="justify-content:center;gap:.3rem">
  {% set base = {'name': name, 'tags': filter_tags or None, 'per': per,
                 'nsfw': '1' if show_nsfw else None} %}
  {% if page > 1 %}
  <a class="secondary" style="padding:.35rem .7rem;border-radius:6px" href="{{ url_for('person_page', page=page-1, **base) }}#fotos">‹ Anterior</a>
  {% else %}<span class="secondary" style="padding:.35rem .7rem;border-radius:6px;opacity:.45">‹ Anterior</span>{% endif %}
  <span class="meta">página <b>{{ page }}</b> de {{ n_pages }}</span>
  {% if page < n_pages %}
  <a class="secondary" style="padding:.35rem .7rem;border-radius:6px" href="{{ url_for('person_page', page=page+1, **base) }}#fotos">Siguiente ›</a>
  {% else %}<span class="secondary" style="padding:.35rem .7rem;border-radius:6px;opacity:.45">Siguiente ›</span>{% endif %}
</div>
{% endif %}
{% endmacro %}

<div class="row">
  <span class="meta">clic en una foto abre el visor (ruedita = zoom, arrastrar = mover, ← → = pasar dentro de esta página).</span>
  {% if nsfw_count %}
    {% if show_nsfw %}
    <a class="secondary" style="padding:.35rem .7rem;border-radius:6px" href="{{ url_for('person_page', name=name, tags=filter_tags or None, per=per) }}#fotos">🙈 Ocultar NSFW ({{ nsfw_count }})</a>
    {% else %}
    <a class="secondary" style="padding:.35rem .7rem;border-radius:6px" href="{{ url_for('person_page', name=name, tags=filter_tags or None, per=per, nsfw=1) }}#fotos">🔞 Mostrar NSFW ({{ nsfw_count }})</a>
    {% endif %}
  {% endif %}
</div>
{{ pager() }}
{% if not display %}
<div class="panel">
  {% if filter_tags %}Ninguna de sus {{ n_todas }} fotos tiene esas etiquetas.
  {% else %}No hay fotos para mostrar.{% endif %}
</div>
{% endif %}
<div class="masonry" id="fotos">
  {% for f in display %}
  <div class="shot" data-tags="{{ f.tags|join(' ') }}">
    <img src="{{ url_for('person_thumb', name=name, rel=f.rel) }}" loading="lazy" alt="{{ f.rel }}"
         onclick="lbOpen({{ loop.index0 }})">
    {% if f.nsfw %}<div class="fn" style="color:var(--red)">🔞 {{ f.rel }}</div>
    {% else %}<div class="fn" title="{{ f.rel }}">{{ f.rel }}</div>{% endif %}
    <button class="primary" onclick="setCover(this, {{ f.rel|tojson }})">⭐ Usar de portada</button>
    <div class="phototags" data-rel="{{ f.rel }}"></div>
    <input class="ptadd" list="tagsug" placeholder="🏷 etiqueta + Enter"
           onkeydown="if(event.key==='Enter'){event.preventDefault();addPhotoTag(this,this.value);this.value=''}">
    <form method="post" action="{{ url_for('person_move_photo', name=name) }}" style="margin-top:.3rem">
      <input type="hidden" name="rel" value="{{ f.rel }}">
      <select name="to" style="width:100%;font-size:.8rem"
              onchange="if (this.value) this.form.requestSubmit()">
        <option value="">🚚 No es {{ name }}… mover a</option>
        <option value="__review__">🗂 Sacar a _revisar</option>
        {% for p in people %}{% if p != name %}<option value="{{ p }}">👤 {{ p }}</option>{% endif %}{% endfor %}
      </select>
    </form>
  </div>
  {% endfor %}
</div>
{{ pager() }}

<datalist id="tagsug">{% for t in all_tags %}<option value="{{ t }}">{% endfor %}</datalist>

<script>
window.LB_PHOTOS = [
  {% for f in display %}{url:"{{ url_for('person_photo', name=name, rel=f.rel) }}",name:"{{ f.rel|e }}",rel:{{ f.rel|tojson }}},{% endfor %}
];
window.POSTS = {{ posts|tojson }};             // {rel: {caption, comments}}
window.VLM_MODELS = {{ vlm_models|tojson }};
window.HAS_COVER = {{ has_thumb|tojson }};
window.COVER_URL_THUMB = "{{ url_for('known_thumb', name=name) }}?v={{ cache_bust }}";
const PERSON = {{ name|tojson }};
let PERSON_TAGS = {{ person_tags|tojson }};
const PHOTO_TAGS = {{ photo_tags|tojson }};   // {rel: [tags propias]}
window.SHORTCUTS = {{ shortcuts|tojson }};    // botones-atajo del modal
const PTAGS_URL = "{{ url_for('tags_person', name=name) }}";
const PHOTOTAG_URL = "{{ url_for('tags_photo', name=name) }}";

function chip(text, onx, inherited) {
  const c = document.createElement('span');
  c.className = 'chip' + (inherited ? ' inherited' : '');
  c.innerHTML = text.replace(/</g,'&lt;') + (inherited ? '' :
    ' <span class="x">×</span>');
  if (!inherited) c.querySelector('.x').onclick = onx;
  return c;
}
function post(url, body) {
  return fetch(url, {method:'POST', headers:{'Content-Type':'application/x-www-form-urlencoded'},
    body: new URLSearchParams(body)});
}

// --- etiquetas de la persona ---
function renderPersonTags() {
  const box = document.getElementById('ptags');
  const input = document.getElementById('ptag-input');
  [...box.querySelectorAll('.chip')].forEach(c => c.remove());
  PERSON_TAGS.forEach(t => box.insertBefore(
    chip(t, () => removePersonTag(t), false), input));
}
function addPersonTag(v) {
  v = v.trim().toLowerCase();
  if (!v || PERSON_TAGS.includes(v)) return;
  PERSON_TAGS.push(v); renderPersonTags();
  post(PTAGS_URL, {tags: PERSON_TAGS.join(',')}).then(refreshAllPhotoTags);
}
function removePersonTag(t) {
  PERSON_TAGS = PERSON_TAGS.filter(x => x !== t); renderPersonTags();
  post(PTAGS_URL, {tags: PERSON_TAGS.join(',')}).then(refreshAllPhotoTags);
}

// --- etiquetas por foto (propias + heredadas de la persona) ---
function renderPhotoTags(div) {
  const rel = div.dataset.rel;
  div.innerHTML = '';
  PERSON_TAGS.forEach(t => div.appendChild(chip(t, null, true)));   // heredadas
  (PHOTO_TAGS[rel] || []).forEach(t => div.appendChild(
    chip(t, () => removePhotoTag(div, t), false)));
}
function refreshAllPhotoTags() {
  document.querySelectorAll('.phototags').forEach(renderPhotoTags);
}
function addPhotoTag(inputEl, v) {
  v = v.trim().toLowerCase();
  const div = inputEl.closest('.shot').querySelector('.phototags');
  const rel = div.dataset.rel;
  if (!v) return;
  const own = PHOTO_TAGS[rel] || (PHOTO_TAGS[rel] = []);
  if (own.includes(v) || PERSON_TAGS.includes(v)) return;   // ya la tiene (propia o heredada)
  own.push(v); renderPhotoTags(div);
  post(PHOTOTAG_URL, {rel: rel, tags: own.join(',')});
}
function removePhotoTag(div, t) {
  const rel = div.dataset.rel;
  PHOTO_TAGS[rel] = (PHOTO_TAGS[rel] || []).filter(x => x !== t);
  renderPhotoTags(div);
  post(PHOTOTAG_URL, {rel: rel, tags: (PHOTO_TAGS[rel]).join(',')});
}

renderPersonTags();
refreshAllPhotoTags();

// --- cómo habla esta persona (va al prompt del pie de foto) ---
const VOICE_URL = "{{ url_for('person_voice', name=name) }}";
function saveVoice() {
  const st = document.getElementById('pvoice-st');
  st.textContent = ' guardando…';
  post(VOICE_URL, {voice: document.getElementById('pvoice').value}).then(r => r.json())
    .then(d => {
      st.textContent = d.ok ? ' ✓ guardado' : ' error';
      setTimeout(() => st.textContent = '', 1800);
    }).catch(() => st.textContent = ' error');
}

// El filtro por etiqueta lo hace el SERVIDOR (el <form> de arriba), no el
// navegador: con la galería paginada, esconder tarjetas del DOM filtraría solo
// la página que estás viendo y te diría "3 fotos" habiendo 40 en las otras.

// --- elegir portada SIN recargar la galería (clave con muchas fotos) ---
const COVER_URL = "{{ url_for('person_cover', name=name) }}";
const COVER_THUMB = "{{ url_for('known_thumb', name=name) }}";
function setCover(btn, rel) {
  const old = btn.textContent;
  btn.textContent = '⏳ …'; btn.disabled = true;
  post(COVER_URL, {rel: rel, ajax: '1'}).then(r => r.json()).then(d => {
    btn.disabled = false; btn.textContent = old;
    if (d.status === 'multi') { window.location = d.pick_url; return; }
    if (d.status === 'error') { alert(d.msg); return; }
    // actualizar la portada de arriba sin recargar (crea el img si aún no había)
    const box = document.querySelector('.cover-box');
    let cov = box.querySelector('img');
    if (!cov) {
      const noimg = box.querySelector('.noimg');
      cov = document.createElement('img');
      if (noimg) noimg.replaceWith(cov); else box.prepend(cov);
    }
    cov.src = COVER_THUMB + '?v=' + Date.now();
    btn.textContent = '✓ portada'; setTimeout(() => btn.textContent = old, 1500);
  }).catch(() => { btn.disabled = false; btn.textContent = old; });
}
</script>
__LIGHTBOX__
{% elif folder %}
<div class="panel">En <b>{{ folder }}</b> no hay imágenes. La galería sale de la carpeta que
  usás para importar conocidos (una subcarpeta por persona).</div>
{% else %}
<div class="panel">Todavía no está configurada la carpeta de las galerías, así que no tengo de
  dónde sacar las fotos de {{ name }}. Elegila con el botón <b>⚙ Configuración</b> de
  <a href="{{ url_for('known_page') }}">Personas conocidas</a> (o importá conocidos desde la
  página principal, que la guarda sola) y volvé acá.</div>
{% endif %}
"""

# --- visor de fotos compartido (galería de persona y revisión NSFW) ---
# Lee window.LB_PHOTOS = [{url, name}, ...] que define cada página.
# --- visor estilo publicación, SOLO para la galería de una persona -----------
# El LIGHTBOX de abajo es el pelado, y lo siguen usando duplicados, NSFW y
# separar-caras: un post de Insta no tiene sentido ahí.
#
# Acá la foto va a la izquierda y a la derecha la columna del post: el perfil
# (portada + nombre), el pie que escribió la IA, los comentarios, y el panel de
# etiquetas que se despliega con "🏷 Etiquetar". Todo sin salir del modal: la
# idea es taguear salteado mientras mirás, sin recorrer 1500 fotos.
POST_LIGHTBOX = """
<div id="lb" class="lb-post">
  <span id="lb-x" onclick="lbClose()" title="Cerrar">✕</span>
  <button class="lb-btn" id="lb-prev" onclick="lbStep(-1)" title="Anterior">‹</button>
  <button class="lb-btn" id="lb-next" onclick="lbStep(1)" title="Siguiente">›</button>
  <div class="post">
    <div class="stage"><img id="lb-img" src="" alt=""></div>
    <aside class="side">
      <header class="phead">
        <img id="lb-av" class="av" src="" alt="">
        <div>
          <b id="lb-who"></b>
          <div class="meta" id="lb-fn"></div>
        </div>
      </header>

      <div class="pbody">
        <div class="pcap" id="lb-cap-box">
          <p id="lb-cap" class="empty">Todavía no tiene pie de foto.</p>
          <div class="row" style="margin:.3rem 0 0">
            <button class="secondary sm" id="lb-cap-btn" onclick="genCaption()">✨ Escribir pie</button>
            <button class="secondary sm" id="lb-cap-edit" onclick="editCap()" title="Reescribirlo a tu manera">✏️</button>
            <span class="meta" id="lb-cap-st"></span>
          </div>
          <textarea id="lb-cap-in" class="capin" style="display:none"
                    placeholder="escribilo a tu manera… (Ctrl+Enter para guardar)"></textarea>
        </div>

        <div class="pcoms">
          <div class="row" style="justify-content:space-between;margin:0 0 .3rem">
            <span class="meta">💬 Comentarios</span>
            <span>
              <button class="secondary sm" id="lb-com-btn" onclick="genComments()">✨ Generar</button>
              <button class="secondary sm" id="lb-com-x" onclick="clearComments()" title="Borrarlos y empezar de nuevo">🗑</button>
            </span>
          </div>
          <div id="lb-coms"></div>
          <span class="meta" id="lb-com-st"></span>
        </div>
      </div>

      <div class="ptags">
        <button class="secondary" id="lb-tag-toggle" onclick="toggleTags()">🏷 Etiquetar ▾</button>
        <div id="lb-tag-panel" style="display:none">
          <span class="meta">Heredadas de {{ name }}:</span>
          <div class="chips" id="lb-inh"></div>
          <span class="meta">Etiquetas de ESTA foto:</span>
          <div class="chips" id="lb-own">
            <input class="taginput" id="lb-tin" list="tagsug" placeholder="escribí y Enter…">
          </div>
          <div class="chips" id="lb-sc" style="margin-top:.4rem"></div>
          <div class="row" style="margin-top:.5rem">
            <button class="secondary sm" id="lb-sug" onclick="suggestTags()">🔮 Sugerir tags</button>
            <span class="meta" id="lb-sug-st"></span>
          </div>
          <div class="chips" id="lb-sugtags" style="margin-top:.3rem"></div>
        </div>
      </div>

      <div class="pfoot">
        <select id="lb-model" title="Con qué modelo escribe la IA">
          {% for m in vlm_models %}<option value="{{ m }}">{{ m }}</option>{% endfor %}
        </select>
        <span class="meta" id="lb-pos"></span>
      </div>
    </aside>
  </div>
</div>
<script>
(function() {
  const P = window.LB_PHOTOS || [], POSTS = window.POSTS || {};
  const lb = document.getElementById('lb'), img = document.getElementById('lb-img');
  const $ = id => document.getElementById(id);
  let idx = 0, scale = 1, tx = 0, ty = 0, drag = false, sx = 0, sy = 0;
  const CAP_URL = "{{ url_for('post_caption', name=name) }}";
  const CAPED_URL = "{{ url_for('post_caption_edit', name=name) }}";
  const COM_URL = "{{ url_for('post_comments', name=name) }}";
  const COMX_URL = "{{ url_for('post_comments_clear', name=name) }}";
  const SUG_URL = "{{ url_for('tags_suggest', name=name) }}";

  const rel = () => P[idx].rel;
  const post_ = (url, body) => fetch(url, {method:'POST',
      headers:{'Content-Type':'application/x-www-form-urlencoded'},
      body: new URLSearchParams(body)}).then(r => r.json());
  const esc = s => String(s).replace(/[<>&]/g, c => ({'<':'&lt;','>':'&gt;','&':'&amp;'}[c]));
  function slot(r) { return POSTS[r] || (POSTS[r] = {caption:null, comments:[]}); }

  // --- zoom y arrastre (igual que el visor pelado) ---
  function apply() { img.style.transform = `translate(${tx}px,${ty}px) scale(${scale})`; }
  function reset() { scale = 1; tx = 0; ty = 0; apply(); }

  function render() {
    const d = slot(rel());
    $('lb-cap').textContent = d.caption || 'Todavía no tiene pie de foto.';
    $('lb-cap').className = d.caption ? '' : 'empty';
    $('lb-cap-btn').textContent = d.caption ? '✨ Reescribir pie' : '✨ Escribir pie';
    $('lb-cap-edit').style.display = d.caption ? '' : 'none';
    cancelCap();
    const box = $('lb-coms');
    box.innerHTML = d.comments.length ? '' : '<span class="meta empty">Sin comentarios.</span>';
    d.comments.forEach(c => {
      const el = document.createElement('div');
      el.className = 'com';
      el.innerHTML = '<b>@' + esc(c.user) + '</b> ' + esc(c.text);
      box.appendChild(el);
    });
    $('lb-com-btn').textContent = d.comments.length ? '✨ Otros' : '✨ Generar';
    $('lb-com-x').style.display = d.comments.length ? '' : 'none';
    $('lb-cap-st').textContent = ''; $('lb-com-st').textContent = '';
    $('lb-sug-st').textContent = ''; $('lb-sugtags').innerHTML = '';
    renderTags();
  }

  function show(i) {
    idx = (i + P.length) % P.length;
    reset();
    img.src = P[idx].url;
    $('lb-fn').textContent = P[idx].name;
    $('lb-pos').textContent = (idx + 1) + ' / ' + P.length;
    render();
  }
  window.lbOpen = function(i) {
    lb.style.display = 'block';
    $('lb-who').textContent = PERSON;
    const av = $('lb-av');
    if (window.HAS_COVER) { av.src = window.COVER_URL_THUMB; av.style.display = ''; }
    else { av.style.display = 'none'; }
    show(i);
  };
  window.lbClose = function() { lb.style.display = 'none'; img.src = ''; };
  window.lbStep = function(d) { show(idx + d); };

  // --- pie de foto ---
  function genCaption() {
    const r = rel(), b = $('lb-cap-btn');
    if (!window.VLM_MODELS.length) {
      $('lb-cap-st').textContent = ' Ollama no está corriendo.'; return;
    }
    b.disabled = true;
    $('lb-cap-st').textContent = ' escribiendo… (tarda ~2 min, es el modelo en CPU)';
    post_(CAP_URL, {rel: r, model: $('lb-model').value}).then(d => {
      b.disabled = false; $('lb-cap-st').textContent = d.ok ? '' : ' ' + d.error;
      if (!d.ok) return;
      slot(r).caption = d.caption;
      if (r === rel()) render();     // ya pasaste de foto: guardalo igual
    }).catch(e => { b.disabled = false; $('lb-cap-st').textContent = ' ' + e; });
  }
  function editCap() {
    const t = $('lb-cap-in');
    t.style.display = ''; t.value = slot(rel()).caption || ''; t.focus();
  }
  function cancelCap() { $('lb-cap-in').style.display = 'none'; }
  $('lb-cap-in').addEventListener('keydown', e => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
      const r = rel(), v = e.target.value.trim();
      post_(CAPED_URL, {rel: r, caption: v}).then(d => {
        if (d.ok) { slot(r).caption = d.caption; render(); }
      });
    } else if (e.key === 'Escape') { e.stopPropagation(); cancelCap(); }
  });

  // --- comentarios ---
  function genComments() {
    const r = rel(), b = $('lb-com-btn');
    if (!window.VLM_MODELS.length) {
      $('lb-com-st').textContent = ' Ollama no está corriendo.'; return;
    }
    b.disabled = true; $('lb-com-st').textContent = ' escribiendo… (~30s)';
    post_(COM_URL, {rel: r, model: $('lb-model').value}).then(d => {
      b.disabled = false; $('lb-com-st').textContent = d.ok ? '' : ' ' + d.error;
      if (!d.ok) return;
      slot(r).comments = d.comments;
      if (r === rel()) render();
    }).catch(e => { b.disabled = false; $('lb-com-st').textContent = ' ' + e; });
  }
  function clearComments() {
    const r = rel();
    post_(COMX_URL, {rel: r}).then(() => { slot(r).comments = []; render(); });
  }

  // --- etiquetas, sin salir del modal ---
  function ownTags() { return (PHOTO_TAGS[rel()] || []).slice(); }
  function saveOwn(tags) {
    const r = rel();
    PHOTO_TAGS[r] = tags;
    post_(PHOTOTAG_URL, {rel: r, tags: tags.join(',')}).then(d => {
      if (d.ok) { PHOTO_TAGS[r] = d.tags; renderTags(); syncGrid(r); }
    });
  }
  function syncGrid(r) {   // que la grilla de atrás no quede desincronizada
    const card = document.querySelector(`.shot .phototags[data-rel="${CSS.escape(r)}"]`);
    if (card && window.renderPhotoTags) renderPhotoTags(card);
  }
  function renderTags() {
    const inh = $('lb-inh');
    inh.innerHTML = PERSON_TAGS.length ? '' : '<span class="meta empty">ninguna</span>';
    PERSON_TAGS.forEach(t => {
      const c = document.createElement('span');
      c.className = 'chip inherited'; c.textContent = t; inh.appendChild(c);
    });
    const box = $('lb-own'), input = $('lb-tin');
    box.querySelectorAll('.chip').forEach(c => c.remove());
    ownTags().forEach(t => {
      const c = document.createElement('span');
      c.className = 'chip';
      c.innerHTML = esc(t) + ' <span class="x">×</span>';
      c.querySelector('.x').onclick = () => saveOwn(ownTags().filter(x => x !== t));
      box.insertBefore(c, input);
    });
    const sc = $('lb-sc');
    sc.innerHTML = '';
    (window.SHORTCUTS || []).forEach(t => {
      const on = ownTags().includes(t);
      const b = document.createElement('button');
      b.className = 'secondary sm' + (on ? ' on' : '');
      b.textContent = (on ? '✓ ' : '+ ') + t;
      b.onclick = () => saveOwn(on ? ownTags().filter(x => x !== t) : ownTags().concat([t]));
      sc.appendChild(b);
    });
  }
  $('lb-tin').addEventListener('keydown', e => {
    if (e.key === 'Enter') {
      e.preventDefault();
      const v = e.target.value.trim();
      if (v && !ownTags().includes(v)) saveOwn(ownTags().concat([v]));
      e.target.value = '';
    } else if (e.key === 'Escape') { e.stopPropagation(); }
  });
  function toggleTags() {
    const p = $('lb-tag-panel'), open = p.style.display === 'none';
    p.style.display = open ? '' : 'none';
    $('lb-tag-toggle').textContent = open ? '🏷 Etiquetar ▴' : '🏷 Etiquetar ▾';
  }
  function suggestTags() {
    const r = rel(), b = $('lb-sug');
    b.disabled = true; $('lb-sug-st').textContent = ' pensando…';
    post_(SUG_URL, {rel: r, engine: 'blip'}).then(d => {
      b.disabled = false; $('lb-sug-st').textContent = d.ok ? '' : ' ' + d.error;
      if (!d.ok || r !== rel()) return;
      const box = $('lb-sugtags');
      box.innerHTML = '';
      d.tags.forEach(t => {
        const b2 = document.createElement('button');
        b2.className = 'secondary sm';
        b2.textContent = '+ ' + t.tag + ' (' + Math.round(t.score * 100) + '%)';
        b2.onclick = () => { if (!ownTags().includes(t.tag)) saveOwn(ownTags().concat([t.tag])); };
        box.appendChild(b2);
      });
    }).catch(e => { b.disabled = false; $('lb-sug-st').textContent = ' ' + e; });
  }

  Object.assign(window, {genCaption, editCap, genComments, clearComments,
                         toggleTags, suggestTags});

  lb.addEventListener('wheel', e => {
    if (!e.target.closest('.stage')) return;    // la columna del post scrollea normal
    e.preventDefault();
    const f = e.deltaY < 0 ? 1.15 : 1 / 1.15;
    const ns = Math.min(8, Math.max(1, scale * f));
    const rc = img.getBoundingClientRect();
    const cx = e.clientX - (rc.left + rc.width / 2), cy = e.clientY - (rc.top + rc.height / 2);
    tx = cx - (cx - tx) * (ns / scale); ty = cy - (cy - ty) * (ns / scale);
    scale = ns;
    if (scale === 1) { tx = 0; ty = 0; }
    apply();
  }, {passive: false});
  img.addEventListener('pointerdown', e => {
    drag = true; sx = e.clientX - tx; sy = e.clientY - ty;
    img.classList.add('grabbing'); img.setPointerCapture(e.pointerId);
  });
  img.addEventListener('pointermove', e => {
    if (drag) { tx = e.clientX - sx; ty = e.clientY - sy; apply(); }
  });
  img.addEventListener('pointerup', () => { drag = false; img.classList.remove('grabbing'); });
  img.addEventListener('dblclick', reset);
  document.querySelector('#lb .stage').addEventListener('click', e => {
    if (e.target.classList.contains('stage')) lbClose();
  });
  document.addEventListener('keydown', e => {
    if (lb.style.display !== 'block') return;
    // escribiendo una etiqueta o el pie: las flechas son para el texto
    const t = e.target.tagName;
    if (t === 'INPUT' || t === 'TEXTAREA') { if (e.key !== 'Escape') return; }
    if (e.key === 'Escape') lbClose();
    else if (e.key === 'ArrowLeft') lbStep(-1);
    else if (e.key === 'ArrowRight') lbStep(1);
  });
})();
</script>
"""

LIGHTBOX = """
<div id="lb">
  <div class="lb-hint">ruedita: zoom · arrastrar: mover · doble clic: reset · ← → · Esc</div>
  <span id="lb-x" onclick="lbClose()" title="Cerrar">✕</span>
  <button class="lb-btn" id="lb-prev" onclick="lbStep(-1)" title="Anterior">‹</button>
  <button class="lb-btn" id="lb-next" onclick="lbStep(1)" title="Siguiente">›</button>
  <div class="stage"><img id="lb-img" src="" alt=""></div>
  <div class="lb-cap" id="lb-cap"></div>
</div>
<script>
(function() {
  const LB_PHOTOS = window.LB_PHOTOS || [];
  const lb = document.getElementById('lb'), img = document.getElementById('lb-img'),
        cap = document.getElementById('lb-cap');
  let idx = 0, scale = 1, tx = 0, ty = 0, drag = false, sx = 0, sy = 0;

  function apply() { img.style.transform = `translate(${tx}px,${ty}px) scale(${scale})`; }
  function reset() { scale = 1; tx = 0; ty = 0; apply(); }
  function show(i) {
    idx = (i + LB_PHOTOS.length) % LB_PHOTOS.length;
    reset();
    img.src = LB_PHOTOS[idx].url;
    cap.textContent = (idx + 1) + ' / ' + LB_PHOTOS.length + ' · ' + LB_PHOTOS[idx].name;
  }
  window.lbOpen = function(i) { lb.style.display = 'block'; show(i); };
  window.lbClose = function() { lb.style.display = 'none'; img.src = ''; };
  window.lbStep = function(d) { show(idx + d); };

  // zoom con ruedita, centrado donde está el puntero
  lb.addEventListener('wheel', e => {
    e.preventDefault();
    const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
    const ns = Math.min(8, Math.max(1, scale * factor));
    const r = img.getBoundingClientRect();
    const cx = e.clientX - (r.left + r.width / 2);
    const cy = e.clientY - (r.top + r.height / 2);
    tx = cx - (cx - tx) * (ns / scale);
    ty = cy - (cy - ty) * (ns / scale);
    scale = ns;
    if (scale === 1) { tx = 0; ty = 0; }
    apply();
  }, {passive: false});

  // arrastre
  img.addEventListener('pointerdown', e => {
    drag = true; sx = e.clientX - tx; sy = e.clientY - ty;
    img.classList.add('grabbing'); img.setPointerCapture(e.pointerId);
  });
  img.addEventListener('pointermove', e => {
    if (!drag) return;
    tx = e.clientX - sx; ty = e.clientY - sy; apply();
  });
  img.addEventListener('pointerup', e => { drag = false; img.classList.remove('grabbing'); });
  img.addEventListener('dblclick', reset);

  // clic en el fondo oscuro (fuera de la imagen) cierra
  document.querySelector('#lb .stage').addEventListener('click', e => {
    if (e.target.classList.contains('stage')) lbClose();
  });

  document.addEventListener('keydown', e => {
    if (lb.style.display !== 'block') return;
    if (e.key === 'Escape') lbClose();
    else if (e.key === 'ArrowLeft') lbStep(-1);
    else if (e.key === 'ArrowRight') lbStep(1);
  });
})();
</script>
"""

# la galería de persona usa el visor estilo publicación; el resto, el pelado
PERSON_TEMPLATE = PERSON_TEMPLATE.replace("__LIGHTBOX__", POST_LIGHTBOX)
# DUPES_TEMPLATE se define antes que LIGHTBOX, así que su visor se enchufa acá
DUPES_TEMPLATE = DUPES_TEMPLATE.replace("__LIGHTBOX__", LIGHTBOX)


@app.route("/person/<name>")
def person_page(name):
    """Galería de una persona conocida: sus fotos de la carpeta importada,
    con elección de portada."""
    cfg = load_config()
    known_dir = get_library(cfg)
    normal = person_photos(known_dir, name) if known_dir else []
    nsfw_list = person_photos(known_dir, name, only_nsfw=True) if known_dir else []
    show_nsfw = request.args.get("nsfw") == "1"
    # lista mostrada: normales, y si el toggle está activo también las NSFW.
    # Cada foto lleva sus etiquetas efectivas (propias + heredadas) para filtrar.
    display = [{"rel": r, "nsfw": False, "tags": tags_mod.effective_photo(name, r)} for r in normal]
    if show_nsfw:
        display += [{"rel": r, "nsfw": True, "tags": tags_mod.effective_photo(name, r)} for r in nsfw_list]
    photos = normal   # para los avisos "no hay imágenes / configurá carpeta"

    # El filtro por etiquetas es del SERVIDOR, no del navegador: si filtrara la
    # página ya paginada te mostraría "3 fotos" cuando en realidad hay 40 en el
    # resto de las páginas.
    filter_tags = request.args.get("tags", "")
    buscadas = [t for t in (_norm_search(t) for t in filter_tags.split(",")) if t]
    if buscadas:
        display = [f for f in display
                   if all(any(t in _norm_search(x) for x in f["tags"]) for t in buscadas)]
    n_filtradas = len(display)

    # Paginado: sin esto, 1500 fotos son 1500 <select> con TODA la gente adentro
    # (decenas de miles de <option>), y el navegador se arrastra.
    per = request.args.get("per", type=int) or cfg.get("gallery_per", PER_DEFAULT)
    per = per if per in PER_OPTIONS else PER_DEFAULT
    if per != cfg.get("gallery_per"):        # la elección queda pegada
        cfg["gallery_per"] = per
        save_config(cfg)
    n_pages = max(1, math.ceil(n_filtradas / per))
    page = min(max(1, request.args.get("page", 1, type=int)), n_pages)
    display = display[(page - 1) * per: page * per]
    thumb = KNOWN_THUMBS / f"{safe_name(name)}.jpg"

    # vecinos en el mismo orden que la galería (para ‹ anterior / siguiente ›)
    people = sorted(load_known(), key=str.lower)
    try:
        i = people.index(name)
    except ValueError:
        i = -1
    prev_name = people[i - 1] if i > 0 else ""
    next_name = people[i + 1] if 0 <= i < len(people) - 1 else ""

    # selector "¿cuál es la cara?": la foto elegida de portada tenía varias
    pick_rel, pick_count = request.args.get("pick", ""), 0
    if pick_rel and known_dir:
        pick_path = Path(known_dir) / name / pick_rel
        if pick_path.is_file():
            try:
                pick_count = save_face_crops(pick_path, PICK_THUMBS)
            except Exception:
                pick_count = 0
    if pick_count < 2:
        pick_rel = ""

    meta = person_meta.get(name)
    # etiquetas propias de cada foto mostrada (para renderizarlas en la grilla)
    photo_tags = {f["rel"]: tags_mod.get_photo(name, f["rel"]) for f in display}
    photo_tags = {k: v for k, v in photo_tags.items() if v}
    return render_template_string(
        shell("known", PERSON_TEMPLATE),
        name=name,
        # el pie y los comentarios ya guardados de cada foto: el modal los
        # muestra al instante, sin esperar al modelo
        posts=posts_mod.get_many(name, [f["rel"] for f in display]),
        # si Ollama no está corriendo, la lista viene vacía y el modal lo avisa
        vlm_models=vlm.list_models(),
        shortcuts=tags_mod.get_shortcuts(),
        people=people,       # para el desplegable "mover a otra persona"
        person_tags=tags_mod.get_person(name),
        photo_tags=photo_tags,
        all_tags=tags_mod.all_tags(),
        filter_tags=filter_tags,       # pre-filtro que viene de la galería
        photos=photos,
        display=display,
        # paginado
        page=page, n_pages=n_pages, per=per, per_options=PER_OPTIONS,
        n_filtradas=n_filtradas,
        n_todas=len(normal) + (len(nsfw_list) if show_nsfw else 0),
        desde=(page - 1) * per + 1 if display else 0,
        hasta=(page - 1) * per + len(display),
        nsfw_count=len(nsfw_list),
        show_nsfw=show_nsfw,
        folder=str(Path(known_dir) / name) if known_dir else "",
        has_thumb=thumb.is_file(),
        cache_bust=int(thumb.stat().st_mtime) if thumb.is_file() else 0,
        prev_name=prev_name,
        next_name=next_name,
        pos=(i + 1) if i >= 0 else 0,
        total=len(people),
        anchor=(i if i >= 0 else 0),
        corrected_ago=person_meta.humanize(meta["corrected"]),
        searched_ago=person_meta.humanize(meta["searched"]),
        voice=meta["voice"],
        pick_rel=pick_rel,
        pick_count=pick_count,
        pick_v=int(time.time()),
        message=request.args.get("msg", ""),
    )


@app.route("/person_photo/<name>/<path:rel>")
def person_photo(name, rel):
    """Sirve la foto ORIGINAL (para el visor con zoom)."""
    cfg = load_config()
    return send_from_directory(Path(get_library(cfg)) / name, rel)


@app.route("/person_thumb/<name>/<path:rel>")
def person_thumb(name, rel):
    """Miniatura cacheada para la GRILLA (rápida). Genera una vez por foto y la
    guarda; en visitas siguientes la sirve del caché. Con loading=lazy, solo se
    generan las que ves al scrollear, no las 1500 de golpe."""
    library = get_library()
    src = Path(library) / name / rel if library else None
    if not src or not src.is_file():
        abort(404)
    key = hashlib.sha1(f"{name}/{rel}".encode("utf-8")).hexdigest() + ".jpg"
    cache = GALLERY_THUMBS / key
    if not cache.is_file() or cache.stat().st_mtime < src.stat().st_mtime:
        GALLERY_THUMBS.mkdir(parents=True, exist_ok=True)
        try:
            im = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
            im.thumbnail((360, 360))
            im.save(cache, quality=82)
        except Exception:
            return send_from_directory(Path(library) / name, rel)  # ilegible: el original
    return send_from_directory(GALLERY_THUMBS, key)


@app.route("/person/<name>/cover", methods=["POST"])
def person_cover(name):
    """Elige la portada. Con una sola cara va directo; con varias redirige al
    selector para que el usuario señale cuál es la persona. La cara confirmada
    se suma a la base como ejemplo de esa persona.

    Con ajax=1 responde JSON (no recarga la galería — clave con 1500 fotos)."""
    cfg = load_config()
    rel = request.form.get("rel", "")
    face = request.form.get("face", "")
    ajax = request.form.get("ajax") == "1"
    path = Path(get_library(cfg)) / name / rel

    def out(status, msg="", pick=None):
        if ajax:
            r = {"status": status, "msg": msg}
            if pick:
                r["pick_url"] = url_for("person_page", name=name, pick=pick)
            return r
        if status == "multi":
            return redirect(url_for("person_page", name=name, pick=pick))
        return redirect(url_for("person_page", name=name, msg=msg))

    if not path.is_file():
        return out("error", "Esa foto ya no está.")
    try:
        status, enc = set_cover(name, path, int(face) if face != "" else None)
    except Exception as e:
        return out("error", f"No se pudo usar esa foto: {e}")

    if status == "multi":
        return out("multi", pick=rel)

    if enc is not None:
        db = load_known()
        add_faces(db, name, [enc])   # cara confirmada -> ejemplo de la persona
        save_known(db)
        msg = f"Portada actualizada y cara sumada como ejemplo de {name}."
    else:
        msg = "Portada actualizada. En esa foto no se detectó una cara, se usa la foto entera."
    return out("ok", msg)


@app.route("/person/<name>/move_photo", methods=["POST"])
def person_move_photo(name):
    """Saca una foto que no es de esta persona: la manda a la carpeta de otra
    persona de la Biblioteca, o la aparta a _revisar."""
    library = get_library()
    rel = request.form.get("rel", "")
    to = request.form.get("to", "")
    if not library or not rel or not to:
        return redirect(url_for("person_page", name=name))

    to_name = None if to == "__review__" else to
    if to_name and to_name not in load_known():
        return redirect(url_for("person_page", name=name, msg="Esa persona no existe."))
    try:
        move_photo(library, name, rel, to_name)
    except Exception as e:
        return redirect(url_for("person_page", name=name, msg=f"No se pudo mover: {e}"))

    # las etiquetas propias de la foto la acompañan a su nuevo lugar
    if to_name:
        tags_mod.move_photo_tags(name, rel, to_name, Path(rel).name)
    # el pie está escrito en la voz de quien la foto YA NO es, así que no sirve
    # en su nuevo dueño: se borra en vez de mudarse
    posts_mod.clear(name, rel)
    destino = f"la carpeta de {to_name}" if to_name else "_revisar"
    return redirect(url_for("person_page", name=name,
                            msg=f"Foto movida a {destino}. (La base de caras no cambió: si ya "
                                f"había aprendido de esa foto, corré el corrector para limpiarla.)"))


def _norm_search(s):
    """Para buscar etiquetas: sin mayúsculas y sin tildes, así "penetracion"
    encuentra "penetración". Igual que normt() del navegador."""
    s = unicodedata.normalize("NFD", (s or "").strip().lower())
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def _photo_path(name, rel):
    """La foto de esa persona en la Biblioteca, o None si ya no está."""
    library = get_library()
    p = Path(library) / name / rel if library else None
    return p if p and p.is_file() else None


@app.route("/post/<name>/caption", methods=["POST"])
def post_caption(name):
    """Escribe el pie de foto con el modelo de visión y lo guarda.

    LENTO: ~2 minutos en CPU. Por eso es a pedido (un botón) y queda guardado:
    la próxima vez que abras la foto sale al toque."""
    rel = request.form.get("rel", "")
    path = _photo_path(name, rel)
    if not path:
        return {"ok": False, "error": "Esa foto ya no está."}
    try:
        text = posts_mod.generate_caption(name, rel, str(path),
                                          model=request.form.get("model") or None)
    except Exception as e:
        return {"ok": False, "error": str(e)}
    return {"ok": True, "caption": text}


@app.route("/post/<name>/caption/edit", methods=["POST"])
def post_caption_edit(name):
    """Reescribe el pie a mano (la IA propone, vos disponés)."""
    rel = request.form.get("rel", "")
    if not rel:
        return {"ok": False}
    return {"ok": True,
            "caption": posts_mod.set_caption(name, rel, request.form.get("caption", ""))}


@app.route("/post/<name>/comments", methods=["POST"])
def post_comments(name):
    """Escribe los comentarios de mentira y los guarda. ~30s.

    El tono sale de Ajustes; los @usuarios de la lista de Ajustes."""
    rel = request.form.get("rel", "")
    path = _photo_path(name, rel)
    if not path:
        return {"ok": False, "error": "Esa foto ya no está."}
    cfg = load_config()
    try:
        got = posts_mod.generate_comments(
            name, rel, str(path),
            tone=cfg.get("comment_tone", posts_mod.DEFAULT_TONE),
            users=cfg.get("comment_users") or None,
            model=request.form.get("model") or None)
    except Exception as e:
        return {"ok": False, "error": str(e)}
    return {"ok": True, "comments": got}


@app.route("/post/<name>/comments/clear", methods=["POST"])
def post_comments_clear(name):
    """Los borra para poder regenerarlos si no gustaron. El pie no se toca."""
    rel = request.form.get("rel", "")
    if rel:
        posts_mod.clear_comments(name, rel)
    return {"ok": True}


@app.route("/person/<name>/voice", methods=["POST"])
def person_voice(name):
    """Guarda cómo habla esta persona: se le suma al prompt de SUS pies de foto.

    Como la foto está en su carpeta sabemos que es ella, así que el pie lo
    escribe ella en primera persona; esto es lo que la hace sonar distinta."""
    return {"ok": True,
            "voice": person_meta.set_voice(name, request.form.get("voice", ""))}


@app.route("/tags/person/<name>", methods=["POST"])
def tags_person(name):
    """Guarda las etiquetas de una persona (texto libre, separadas por coma)."""
    tags = [t for t in request.form.get("tags", "").split(",")]
    saved = tags_mod.set_person(name, tags)
    return {"ok": True, "tags": saved}


@app.route("/tags/photo/<name>", methods=["POST"])
def tags_photo(name):
    """Guarda las etiquetas propias de una foto de esa persona."""
    rel = request.form.get("rel", "")
    if not rel:
        return {"ok": False}
    tags = [t for t in request.form.get("tags", "").split(",")]
    saved = tags_mod.set_photo(name, rel, tags)
    return {"ok": True, "tags": saved}


PHRASES_TEMPLATE = """
<h1>✎ Frases</h1>
<p class="hint">Las frases que escribe el descriptor son formularias y suenan robóticas
   ("una imagen de una mujer acostada en una cama"). Acá las reescribís <b>a tu manera, una
   sola vez</b>: la próxima vez que una foto dé esa misma frase, se muestra la tuya.
   Como se repiten muchísimo, arreglás cientos de fotos con unas pocas ediciones.</p>

{% if message %}<div class="msg">{{ message }}</div>{% endif %}

<div class="panel">
  <div class="counter">
    {% if kind == 'nsfw' %}🔞 NSFW: {% elif kind == 'sfw' %}Sin NSFW: {% endif %}
    {{ st.total }} frase(s) · <b style="color:var(--grn)">{{ st.hechas }}</b> reescritas ·
    {{ st.pendientes }} pendientes</div>
  <div class="row">
    <span class="meta">Ver:</span>
    <a class="secondary {% if kind == 'all' %}primary{% endif %}" style="padding:.35rem .7rem;border-radius:6px"
       href="{{ url_for('phrases_page', pend=1 if pend else None) }}">Todas</a>
    <a class="secondary {% if kind == 'nsfw' %}primary{% endif %}" style="padding:.35rem .7rem;border-radius:6px"
       href="{{ url_for('phrases_page', kind='nsfw', pend=1 if pend else None) }}">🔞 Solo NSFW ({{ st.nsfw_total }})</a>
    <a class="secondary {% if kind == 'sfw' %}primary{% endif %}" style="padding:.35rem .7rem;border-radius:6px"
       href="{{ url_for('phrases_page', kind='sfw', pend=1 if pend else None) }}">Sin NSFW</a>
    <span style="width:1rem"></span>
    <a class="secondary {% if not pend %}primary{% endif %}" style="padding:.35rem .7rem;border-radius:6px"
       href="{{ url_for('phrases_page', kind=kind if kind != 'all' else None) }}">Todas</a>
    <a class="secondary {% if pend %}primary{% endif %}" style="padding:.35rem .7rem;border-radius:6px"
       href="{{ url_for('phrases_page', kind=kind if kind != 'all' else None, pend=1) }}">Solo pendientes</a>
  </div>
  <span class="meta">Ordenadas por las que más aparecieron (esas rinden más). Las de fotos en
    <code>_nsfw</code> se marcan solas; con el 🔞 corregís a mano.</span>
</div>

{% if not items %}
<div class="panel"><p class="meta">
  {% if pend %}✅ No quedan frases pendientes{% if kind == 'nsfw' %} en la lista NSFW{% elif kind == 'sfw' %} fuera de NSFW{% endif %}.
  {% elif kind == 'nsfw' %}Todavía no hay frases marcadas como NSFW. Se marcan solas cuando sugerís
  sobre fotos que están en una carpeta <code>_nsfw</code>, o con el 🔞 a mano.
  {% else %}Todavía no hay frases. Usá "🔮 Sugerir" en el etiquetador y van apareciendo acá solas.{% endif %}
</p></div>
{% else %}
{% for p in items %}
<div class="panel">
  <div class="row" style="justify-content:space-between">
    <span class="meta">apareció <b>{{ p.count }}</b> vez(ces)</span>
    <span class="row" style="margin:0">
      <form method="post" action="{{ url_for('phrases_nsfw') }}" style="margin:0">
        <input type="hidden" name="auto" value="{{ p.auto }}">
        <input type="hidden" name="value" value="{{ 0 if p.nsfw else 1 }}">
        <input type="hidden" name="kind" value="{{ kind }}">
        <input type="hidden" name="pend" value="{{ 1 if pend else '' }}">
        <button style="padding:.1rem .5rem;font-size:.8rem{% if p.nsfw %};background:var(--red);color:#fff;border:0{% endif %}"
                title="{% if p.nsfw %}Sacar de la lista NSFW{% else %}Marcar como NSFW{% endif %}">🔞</button>
      </form>
      <form method="post" action="{{ url_for('phrases_forget') }}" style="margin:0"
            onsubmit="return confirm('¿Sacar esta frase del listado?')">
        <input type="hidden" name="auto" value="{{ p.auto }}">
        <input type="hidden" name="kind" value="{{ kind }}">
        <input type="hidden" name="pend" value="{{ 1 if pend else '' }}">
        <button style="padding:.1rem .5rem;font-size:.8rem">🗑</button>
      </form>
    </span>
  </div>
  <div class="meta" style="font-style:italic; margin:.2rem 0 .4rem">“{{ p.auto }}”</div>
  <form method="post" action="{{ url_for('phrases_save') }}">
    <input type="hidden" name="auto" value="{{ p.auto }}">
    <input type="hidden" name="kind" value="{{ kind }}">
    <input type="hidden" name="pend" value="{{ 1 if pend else '' }}">
    <div class="row">
      <input class="path" type="text" name="custom" value="{{ p.custom }}"
             placeholder="escribila a tu manera… (vacío = usa la automática)">
      <button class="primary">💾 Guardar</button>
      {% if p.custom %}<span class="meta" style="color:var(--grn)">✓ reescrita</span>{% endif %}
    </div>
  </form>
</div>
{% endfor %}
{% endif %}
"""


def _phrase_filters():
    """(kind, pend, nsfw) del querystring/form: kind = all | nsfw | sfw."""
    src = request.form if request.method == "POST" else request.args
    kind = src.get("kind", "all")
    if kind not in ("all", "nsfw", "sfw"):
        kind = "all"
    pend = src.get("pend") == "1"
    nsfw = {"nsfw": True, "sfw": False}.get(kind)   # None = todas
    return kind, pend, nsfw


@app.route("/phrases")
def phrases_page():
    kind, pend, nsfw = _phrase_filters()
    return render_template_string(
        shell("phrases", PHRASES_TEMPLATE),
        items=phrases_mod.list_all(only_pending=pend, nsfw=nsfw),
        st=phrases_mod.stats(nsfw=nsfw),
        kind=kind, pend=pend, message=request.args.get("msg", ""))


def _back_to_phrases(msg):
    """Vuelve a la lista conservando el filtro en el que estabas."""
    kind, pend, _ = _phrase_filters()
    return redirect(url_for("phrases_page", msg=msg,
                            kind=None if kind == "all" else kind,
                            pend=1 if pend else None))


@app.route("/phrases/save", methods=["POST"])
def phrases_save():
    custom = request.form.get("custom", "")
    phrases_mod.set_custom(request.form.get("auto", ""), custom)
    return _back_to_phrases("Frase reescrita." if custom.strip()
                            else "Frase vuelta a la automática.")


@app.route("/phrases/nsfw", methods=["POST"])
def phrases_nsfw():
    """Marca/desmarca una frase como NSFW (para editar esa lista aparte)."""
    val = request.form.get("value") == "1"
    phrases_mod.set_nsfw(request.form.get("auto", ""), val)
    return _back_to_phrases("Marcada como NSFW." if val else "Sacada de NSFW.")


@app.route("/phrases/forget", methods=["POST"])
def phrases_forget():
    phrases_mod.forget(request.form.get("auto", ""))
    return _back_to_phrases("Frase sacada del listado.")


@app.route("/phrases/quick", methods=["POST"])
def phrases_quick():
    """Corregir una frase desde el etiquetador, sin ir a la pantalla de Frases."""
    auto = request.form.get("auto", "")
    custom = request.form.get("custom", "")
    if not auto:
        return {"ok": False}
    return {"ok": True, "custom": phrases_mod.set_custom(auto, custom)}


@app.route("/tags/shortcuts", methods=["POST"])
def tags_shortcuts():
    """Guarda los botones-atajo del etiquetador (editables por el usuario)."""
    sc = [t for t in request.form.get("shortcuts", "").split(",")]
    return {"ok": True, "shortcuts": tags_mod.set_shortcuts(sc)}


@app.route("/tags/suggest/<name>", methods=["POST"])
def tags_suggest(name):
    """Corre el interrogator sobre una foto: devuelve una frase descriptiva y,
    de TU vocabulario (etiquetas + atajos), las que CLIP cree que aplican. No
    etiqueta solo: son sugerencias para que confirmes."""
    library = get_library()
    rel = request.form.get("rel", "")
    path = Path(library) / name / rel if library else None
    if not path or not path.is_file():
        return {"ok": False, "error": "Esa foto ya no está."}
    vocab = sorted(set(tags_mod.all_tags()) | set(tags_mod.get_shortcuts()))
    # motor: "blip" (rápido, incluido) o "vlm" (modelo de visión en Ollama)
    engine = request.form.get("engine", "blip")
    try:
        r = interrogator.interrogate(str(path), vocab, engine=engine,
                                     vlm_model=request.form.get("vlm_model") or None,
                                     person=name)
    except Exception as e:
        return {"ok": False, "error": f"No se pudo analizar: {e}"}
    # la frase automática se registra y, si vos la reescribiste, se usa la tuya.
    # Si la foto está en una carpeta _nsfw, la frase se marca sola como NSFW
    # (así después editás esa lista aparte).
    auto = r["caption"]
    es_nsfw = NSFW_DIR_NAME in Path(rel).parts
    return {"ok": True,
            "caption": phrases_mod.apply(auto, nsfw=es_nsfw),   # tu versión si existe
            "auto": auto,                          # la automática (clave para editarla)
            "details": [{"label": l, "value": v} for l, v in r.get("details", [])],
            "tags": [{"tag": t, "score": round(s, 3)} for t, s in r["tags"]]}


# La búsqueda por etiqueta ya no es una sección aparte: está integrada en la
# Galería (el buscador filtra nombres Y etiquetas) y adentro de cada perfil.


@app.route("/pick_thumb/<fn>")
def pick_thumb(fn):
    return send_from_directory(PICK_THUMBS, fn)


CORRECT_TEMPLATE = """
<style>
  .fn { font-size:.8rem; color:var(--mut); overflow:hidden; text-overflow:ellipsis;
        white-space:nowrap; max-width:480px; display:inline-block; vertical-align:middle; }
</style>
<h1>🧹 Corregir a {{ name }}</h1>
<p><a href="{{ url_for('person_page', name=name) }}">← volver a {{ name }}</a></p>
{% if message %}<div class="msg">{{ message }}</div>{% endif %}
{% if state.status == 'error' %}<div class="err">{{ state.error }}</div>{% endif %}

{% if state.status == 'running' %}
<div class="panel">
  <p>Revisando <b>{{ state.current }}/{{ state.total }}</b>: {{ state.photo }}</p>
  <progress value="{{ state.current }}" max="{{ state.total or 1 }}"></progress>
  <p class="meta">Esta página se actualiza sola cada 2 segundos.</p>
</div>
{% elif state.status == 'done' and data %}
<div class="panel">
  <div class="counter">{{ data.auto|length }} foto(s) obvias confirmadas solas ·
    {{ data.review|length }} dudosa(s) para revisar abajo
    {% if data.no_face %} · {{ data.no_face|length }} sin caras (no entrenan){% endif %}</div>
  <p class="meta">Las obvias tienen una sola cara que encaja con la que más se repite en la carpeta.
     En las dudosas, la verde es la que creo que es {{ name }} (ya viene marcada — corregí solo si me equivoco).
     Si {{ name }} está en la foto pero su cara no aparece entre los recortes (de espaldas, tapada,
     borrosa), usá la opción amarilla: la foto queda en su carpeta, solo que no entrena.
     Las fotos sin ninguna cara detectada tampoco entrenan y se quedan donde están.
     Una foto sin respuesta no entrena ni se mueve.</p>
  {% if data.auto %}
  <button type="button" onclick="const g=document.getElementById('auto-grid'); g.style.display = g.style.display==='none' ? 'flex' : 'none'">
    👀 Ver las {{ data.auto|length }} obvias (solo para chequear)
  </button>
  <div class="faces" id="auto-grid" style="display:none; margin-top:.75rem">
    {% for a in data.auto %}
    <div class="opt" title="{{ a.rel }} — clic para ver la foto completa">
      <img src="{{ url_for('correct_thumb', fn=a.thumb) }}?v={{ v }}" alt="cara"
           data-full="{{ url_for('person_photo', name=name, rel=a.rel) }}"
           onclick="showFull(this.dataset.full)">
    </div>
    {% endfor %}
  </div>
  {% endif %}
</div>

<form method="post" action="{{ url_for('correct_apply', name=name) }}"
      onsubmit="return confirm('Se reemplazan TODOS los ejemplos de {{ name }} por las caras verificadas. ¿Aplicar?')">
  {% for r in data.review %}
  {% set oi = loop.index0 %}
  <div class="panel">
    <div><b>¿Cuál es {{ name }}?</b>
      <a class="fn" href="{{ url_for('person_photo', name=name, rel=r.rel) }}" target="_blank"
         data-full="{{ url_for('person_photo', name=name, rel=r.rel) }}"
         onclick="showFull(this.dataset.full); return false" title="{{ r.rel }}">{{ r.rel }}</a>
      <span class="meta">— clic en un recorte muestra la foto completa; el circulito elige</span></div>
    <div class="faces" style="margin-top:.5rem">
      {% for f in r.faces %}
      <label class="opt {% if f.sug %}sug{% endif %}"
             title="{% if f.sug %}me parece que es {{ name }}{% else %}cara de la foto{% endif %} — clic para ver la foto completa">
        <input type="radio" name="pick-{{ oi }}" value="{{ f.j }}" {% if r.pre == f.j %}checked{% endif %}>
        <img src="{{ url_for('correct_thumb', fn=f.thumb) }}?v={{ v }}" alt="cara"
             data-full="{{ url_for('person_photo', name=name, rel=r.rel) }}"
             onclick="event.preventDefault(); showFull(this.dataset.full)">
      </label>
      {% endfor %}
      <label class="opt none stay-box">
        <input type="radio" name="pick-{{ oi }}" value="skip">
        <span>😶 Está, pero su cara<br>no salió en estos recortes<br><small>(no entrena; la foto queda)</small></span>
      </label>
      <label class="opt none none-box">
        <input type="radio" name="pick-{{ oi }}" value="none">
        <span>🚫 {{ name }} no está<br>en esta foto<br><small>(se aparta a _revisar)</small></span>
      </label>
    </div>
  </div>
  {% endfor %}
  <div class="panel">
    <label><input type="checkbox" name="move_out" value="1" checked>
      Mover a <code>_revisar</code> las fotos donde {{ name }} no está (quedan apartadas, no se borran)</label>
    <div style="margin-top:.75rem">
      <button class="organize">💾 Aplicar corrección: {{ name }} queda solo con caras verificadas</button>
    </div>
  </div>
</form>
{% else %}
<div class="panel">No hay ninguna corrección en curso.
  Arrancala desde la página de <a href="{{ url_for('person_page', name=name) }}">{{ name }}</a>.</div>
{% endif %}

<div id="fmodal-bg" onclick="hideFull()">
  <span id="fmodal-x" title="Cerrar">✕</span>
  <img id="fmodal-img" src="" alt="foto completa">
</div>
<script>
function showFull(url) {
  document.getElementById('fmodal-img').src = url;
  document.getElementById('fmodal-bg').style.display = 'flex';
}
function hideFull() {
  document.getElementById('fmodal-bg').style.display = 'none';
  document.getElementById('fmodal-img').src = '';
}
document.addEventListener('keydown', e => { if (e.key === 'Escape') hideFull(); });
</script>
"""


def _correct_worker(folder):
    def progress(current, total, photo):
        CORRECT_STATE.update(current=current, total=total, photo=photo)
    try:
        data = scan_person_folder(folder, progress=progress)
        # cara preseleccionada de cada dudosa: la primera del grupo dominante
        for r in data["review"]:
            r["pre"] = next((f["j"] for f in r["faces"] if f.get("sug")), None)
        CORRECT_STATE.update(status="done", data=data)
    except Exception as e:
        CORRECT_STATE.update(status="error", error=str(e))


@app.route("/person/<name>/correct/start", methods=["POST"])
def correct_start(name):
    if CORRECT_STATE["status"] == "running":
        return redirect(url_for("correct_page", name=CORRECT_STATE["name"]))
    known_dir = get_library()
    folder = Path(known_dir) / name if known_dir else None
    if not folder or not folder.is_dir():
        return redirect(url_for("person_page", name=name,
                                msg="No hay carpeta de esa persona para revisar."))
    CORRECT_STATE.update(status="running", current=0, total=0, photo="",
                         error="", name=name, folder=str(folder), data=None)
    threading.Thread(target=_correct_worker, args=(folder,), daemon=True).start()
    return redirect(url_for("correct_page", name=name))


@app.route("/person/<name>/correct")
def correct_page(name):
    state = CORRECT_STATE if CORRECT_STATE["name"] == name else \
        {"status": "idle", "current": 0, "total": 0, "photo": "", "error": ""}
    return render_template_string(
        shell("known", CORRECT_TEMPLATE, head_extra=REFRESH_STATE),
        name=name,
        state=state,
        data=state.get("data"),
        v=int(time.time()),
        message=request.args.get("msg", ""),
    )


@app.route("/person/<name>/correct/apply", methods=["POST"])
def correct_apply(name):
    st = CORRECT_STATE
    if st["status"] != "done" or st["name"] != name or not st["data"]:
        return redirect(url_for("person_page", name=name))
    answers = {}
    for i, r in enumerate(st["data"]["review"]):
        v = request.form.get(f"pick-{i}", "")
        if v != "":
            answers[r["rel"]] = v
    try:
        n, moved, errors = apply_correction(
            name, st["folder"], st["data"]["auto"], st["data"]["review"],
            answers, move_out=request.form.get("move_out") == "1")
    except ValueError as e:
        return redirect(url_for("correct_page", name=name, msg=str(e)))
    CORRECT_STATE.update(status="idle", name="", folder="", data=None)
    person_meta.stamp(name, "corrected")
    msg = f"Listo: {name} quedó con {n} ejemplo(s) verificados."
    if moved:
        msg += f" {len(moved)} foto(s) donde no está fueron a _revisar."
    if errors:
        msg += f" {len(errors)} no se pudieron mover."
    return redirect(url_for("person_page", name=name, msg=msg))


@app.route("/correct_thumb/<fn>")
def correct_thumb(fn):
    return send_from_directory(CORRECT_THUMBS, fn)


NSFW_TEMPLATE = """
<h1>Revisar NSFW</h1>
<p class="hint">Revisa una carpeta buscando fotos con <b>desnudez explícita</b> (todo local, no sale nada
   de esta máquina). Solo marca las clases inequívocas — una foto normal de playa no debería saltar.
   Las sospechosas se muestran para que las revises: las que confirmes se <b>mueven</b> a una subcarpeta
   <code>{{ nsfw_dir }}</code> dentro de la carpeta revisada. No se borra nada.</p>

{% if message %}<div class="msg">{{ message }}</div>{% endif %}
{% if state.status == 'error' %}<div class="err">{{ state.error }}</div>{% endif %}

<div class="panel">
  <form method="post" action="{{ url_for('nsfw_scan') }}">
    <div class="row">
      <label><b>Carpeta:</b></label>
      <input class="path" type="text" name="folder" value="{{ state.folder }}"
             placeholder="Carpeta a revisar (incluye subcarpetas)">
      <button type="button" onclick="browse('folder')">📂 Elegir carpeta…</button>
    </div>
    <div class="row">
      <label>Sensibilidad:</label>
      <span class="meta">estricto</span>
      <input type="range" name="sens" min="0" max="100" value="{{ sens }}">
      <span class="meta">sensible</span>
      <span class="meta">(subilo hacia "sensible" si te dejó afuera fotos que son NSFW; bajalo si te trae de más)</span>
    </div>
    <div class="row">
      <label></label>
      <label style="font-weight:400"><input type="checkbox" name="covered" {% if state.covered %}checked{% endif %}>
        Incluir insinuante / ropa interior (bikini, lencería) — atrapa más, más falsos positivos de playa</label>
    </div>
    <button class="primary" {% if state.status == 'running' %}disabled{% endif %}>
      🔞 Revisar la carpeta
    </button>
    <span class="meta">La primera vez tarda un poco más (carga el modelo detector).</span>
  </form>
  {% if state.status == 'running' %}
    <p>Revisando <b>{{ state.current }}/{{ state.total }}</b>: {{ state.photo }}</p>
    <progress value="{{ state.current }}" max="{{ state.total or 1 }}"></progress>
    <form method="post" action="{{ url_for('nsfw_stop') }}" style="margin-top:.5rem">
      <button class="danger">⛔ Detener</button>
    </form>
    <p class="meta">Esta página se actualiza sola cada 2 segundos.</p>
  {% endif %}
</div>

{% if state.status == 'done' %}
<div class="panel">
  <div class="counter">{{ state.results|length }} sospechosa(s) en {{ state.scanned }} foto(s) revisadas</div>
  {% if state.results %}
  <form method="post" action="{{ url_for('nsfw_move') }}"
        onsubmit="return confirm('Se moverán las fotos tildadas a {{ nsfw_dir }} dentro de la carpeta revisada. ¿Continuar?')">
    <div class="row">
      <button class="organize">📦 Mover las tildadas a {{ nsfw_dir }}</button>
      <button type="button" class="secondary" onclick="toggleAll(this)">Marcar / desmarcar todas</button>
      <span class="meta">destildá los falsos positivos (playa, etc.) antes de mover — clic en la imagen abre el visor (zoom, arrastre, ← →)</span>
    </div>
    <div class="grid">
      {% for r in state.results %}
      <label class="hit">
        <input type="checkbox" name="rel" value="{{ r.rel }}" checked>
        <span class="cf">{{ (r.score * 100)|int }}% · {{ r.labels|join(', ') }}</span><br>
        {% if r.thumb %}
        <img src="{{ url_for('nsfw_thumb', fn=r.thumb) }}" alt=""
             onclick="event.preventDefault(); lbOpen({{ loop.index0 }})">
        {% endif %}
        <div class="fn" title="{{ r.rel }}">
          <a href="{{ url_for('nsfw_file', rel=r.rel) }}" target="_blank">{{ r.rel.split('/')[-1] }}</a>
        </div>
      </label>
      {% endfor %}
    </div>
  </form>
  <script>
  window.LB_PHOTOS = [
    {% for r in state.results %}{url:"{{ url_for('nsfw_file', rel=r.rel) }}",name:"{{ r.rel|e }}"},{% endfor %}
  ];
  </script>
  __LIGHTBOX__
  {% else %}
  <p class="meta">✅ Ninguna foto con desnudez explícita en esa carpeta.</p>
  {% endif %}
</div>
{% endif %}

<script>
function browse(field) {
  fetch('{{ url_for('browse') }}')
    .then(r => r.json())
    .then(d => { if (d.path) document.getElementsByName(field)[0].value = d.path; });
}
function toggleAll(btn) {
  const boxes = [...btn.closest('form').querySelectorAll('.grid input[type=checkbox]')];
  const all = boxes.every(b => b.checked);
  boxes.forEach(b => b.checked = !all);
}
</script>
"""

# el visor compartido también en la revisión NSFW
NSFW_TEMPLATE = NSFW_TEMPLATE.replace("__LIGHTBOX__", LIGHTBOX)


def _nsfw_worker(folder, score_min, covered):
    def progress(current, total, photo):
        NSFW_STATE.update(current=current, total=total, photo=photo)
    def should_stop():
        return NSFW_STATE.get("cancel", False)
    try:
        results, scanned = scan_nsfw(folder, score_min=score_min, include_covered=covered,
                                     progress=progress, should_stop=should_stop)
        NSFW_STATE.update(status="done", results=results, scanned=scanned)
    except Exception as e:
        NSFW_STATE.update(status="error", error=str(e))


def _sens_from_score(score_min):
    """score_min (0.60 estricto … 0.20 sensible) -> posición 0-100 del deslizador."""
    frac = (SCORE_MIN_HI - score_min) / (SCORE_MIN_HI - SCORE_MIN_LO)
    return int(round(max(0, min(1, frac)) * 100))


@app.route("/nsfw")
def nsfw_page():
    return render_template_string(
        shell("nsfw", NSFW_TEMPLATE, head_extra=REFRESH_STATE),
        state=NSFW_STATE, nsfw_dir=NSFW_DIR_NAME,
        sens=_sens_from_score(NSFW_STATE.get("score_min", SCORE_MIN)),
        message=request.args.get("msg", ""))


@app.route("/nsfw/scan", methods=["POST"])
def nsfw_scan():
    if NSFW_STATE["status"] == "running":
        return redirect(url_for("nsfw_page"))
    folder = request.form.get("folder", "").strip()
    if not folder or not Path(folder).is_dir():
        return redirect(url_for("nsfw_page", msg="Esa carpeta no existe."))
    # sensibilidad 0-100 -> umbral de confianza (más sensible = umbral más bajo)
    try:
        sens = max(0, min(100, int(request.form.get("sens", "50"))))
    except ValueError:
        sens = 50
    score_min = SCORE_MIN_HI - (sens / 100) * (SCORE_MIN_HI - SCORE_MIN_LO)
    covered = request.form.get("covered") == "on"
    NSFW_STATE.update(status="running", current=0, total=0, photo="", error="",
                      results=[], folder=folder, scanned=0, cancel=False,
                      score_min=score_min, covered=covered)
    threading.Thread(target=_nsfw_worker, args=(folder, score_min, covered), daemon=True).start()
    return redirect(url_for("nsfw_page"))


@app.route("/nsfw/stop", methods=["POST"])
def nsfw_stop():
    if NSFW_STATE["status"] == "running":
        NSFW_STATE["cancel"] = True
    return redirect(url_for("nsfw_page"))


@app.route("/nsfw/move", methods=["POST"])
def nsfw_move():
    if NSFW_STATE["status"] != "done" or not NSFW_STATE["results"]:
        return redirect(url_for("nsfw_page"))
    rels = request.form.getlist("rel")
    if not rels:
        return redirect(url_for("nsfw_page", msg="No había nada tildado."))
    moved, errors = move_to_nsfw(NSFW_STATE["folder"], rels)
    done = set(moved)
    NSFW_STATE["results"] = [r for r in NSFW_STATE["results"] if r["rel"] not in done]
    msg = f"{len(moved)} foto(s) movidas a {NSFW_DIR_NAME}."
    if errors:
        msg += f" {len(errors)} no se pudieron mover (ej: {errors[0][0]})."
    return redirect(url_for("nsfw_page", msg=msg))


@app.route("/nsfw_thumb/<fn>")
def nsfw_thumb(fn):
    return send_from_directory(NSFW_THUMBS, fn)


@app.route("/nsfw_file/<path:rel>")
def nsfw_file(rel):
    return send_from_directory(NSFW_STATE["folder"], rel)


SPLIT_TEMPLATE = """
<h1>Separar caras</h1>
<p class="hint">Encuentra las fotos donde hay <b>más de una persona</b> y las aparta a la subcarpeta
   <code>{{ group_dir }}</code>. ¿Para qué? El importador aprende de cada foto <i>la cara más grande</i>:
   eso es infalible si la persona está sola, pero en una foto grupal puede aprender la cara equivocada.
   Separando las grupales, el import solo aprende de fotos seguras — y las grupales las procesás
   aparte con el <b>corrector</b>, que pregunta cuál cara es cuál.</p>
<p class="hint"><b>Ojo con el tamaño mínimo:</b> solo se cuentan las caras grandes. Así una selfie en la
   calle con transeúntes de fondo <b>no</b> se toma como foto grupal.</p>

{% if message %}<div class="msg">{{ message }}</div>{% endif %}
{% if state.status == 'error' %}<div class="err">{{ state.error }}</div>{% endif %}

<div class="panel">
  <form method="post" action="{{ url_for('split_scan') }}">
    <div class="row">
      <label><b>Carpeta:</b></label>
      <input class="path" type="text" name="folder" value="{{ state.folder }}"
             placeholder="Carpeta a revisar (incluye subcarpetas)">
      <button type="button" onclick="browse('folder')">📂 Elegir carpeta…</button>
    </div>
    <div class="row">
      <label>Tamaño mín. de cara:</label>
      <select name="min_px">
        <option value="48" {% if state.min_px == 48 %}selected{% endif %}>Chico (48px) — cuenta hasta caras lejanas</option>
        <option value="80" {% if state.min_px == 80 %}selected{% endif %}>Medio (80px) — recomendado</option>
        <option value="120" {% if state.min_px == 120 %}selected{% endif %}>Grande (120px) — ignora casi todo el fondo</option>
        <option value="160" {% if state.min_px == 160 %}selected{% endif %}>Muy grande (160px) — solo caras bien en primer plano</option>
      </select>
      <span class="meta">más grande = menos transeúntes contados como "acompañante"</span>
    </div>
    <button class="primary" {% if state.status == 'running' %}disabled{% endif %}>
      ⚇ Buscar fotos con más de una cara
    </button>
  </form>
  {% if state.status == 'running' %}
    <p>Revisando <b>{{ state.current }}/{{ state.total }}</b>: {{ state.photo }}</p>
    <progress value="{{ state.current }}" max="{{ state.total or 1 }}"></progress>
    <form method="post" action="{{ url_for('split_stop') }}" style="margin-top:.5rem">
      <button class="danger">⛔ Detener</button>
    </form>
    <p class="meta">Esta página se actualiza sola cada 2 segundos.</p>
  {% endif %}
</div>

{% if state.status == 'done' %}
{% set r = state.resumen %}
<div class="panel">
  <div class="counter">{{ r.escaneadas }} foto(s) revisadas</div>
  <p class="meta">
    👤 <b>{{ r.solas }}</b> con una sola cara (quedan donde están — el import aprende de estas) ·
    👥 <b>{{ r.grupales }}</b> con dos o más caras (candidatas a <code>{{ group_dir }}</code>) ·
    🚫 <b>{{ r.sin_cara }}</b> sin caras (no se tocan)
  </p>
</div>
{% if state.results %}
<div class="panel">
  <form method="post" action="{{ url_for('split_move') }}"
        onsubmit="return confirm('Se moverán las fotos tildadas a {{ group_dir }} dentro de la carpeta revisada. ¿Continuar?')">
    <div class="row">
      <button class="organize">📦 Mover las tildadas a {{ group_dir }}</button>
      <button type="button" class="secondary" onclick="toggleAll(this)">Marcar / desmarcar todas</button>
      <span class="meta">destildá las que en realidad son de una sola persona — clic en la imagen abre el visor</span>
    </div>
    <div class="grid">
      {% for g in state.results %}
      <label class="hit">
        <input type="checkbox" name="rel" value="{{ g.rel }}" checked>
        <span class="cf">👥 {{ g.faces }} caras</span><br>
        {% if g.thumb %}
        <img src="{{ url_for('split_thumb', fn=g.thumb) }}" loading="lazy" alt=""
             onclick="event.preventDefault(); lbOpen({{ loop.index0 }})">
        {% endif %}
        <div class="fn" title="{{ g.rel }}">{{ g.rel.split('/')[-1] }}</div>
      </label>
      {% endfor %}
    </div>
  </form>
  <script>
  window.LB_PHOTOS = [
    {% for g in state.results %}{url:"{{ url_for('split_file', rel=g.rel) }}",name:"{{ g.rel|e }} ({{ g.faces }} caras)"},{% endfor %}
  ];
  </script>
  __LIGHTBOX__
</div>
{% else %}
<div class="panel"><p class="meta">✅ No hay fotos con más de una cara (con ese tamaño mínimo).
   Podés importar esta carpeta sin riesgo.</p></div>
{% endif %}
{% endif %}

<script>
function browse(field) {
  fetch('{{ url_for('browse') }}')
    .then(r => r.json())
    .then(d => { if (d.path) document.getElementsByName(field)[0].value = d.path; });
}
function toggleAll(btn) {
  const boxes = [...btn.closest('form').querySelectorAll('.grid input[type=checkbox]')];
  const all = boxes.every(b => b.checked);
  boxes.forEach(b => b.checked = !all);
}
</script>
"""

SPLIT_TEMPLATE = SPLIT_TEMPLATE.replace("__LIGHTBOX__", LIGHTBOX)


def _split_worker(folder, min_px):
    def progress(current, total, photo):
        SPLIT_STATE.update(current=current, total=total, photo=photo)
    def should_stop():
        return SPLIT_STATE.get("cancel", False)
    try:
        results, resumen = scan_groups(folder, min_px=min_px,
                                       progress=progress, should_stop=should_stop)
        SPLIT_STATE.update(status="done", results=results, resumen=resumen)
    except Exception as e:
        SPLIT_STATE.update(status="error", error=str(e))


@app.route("/split")
def split_page():
    return render_template_string(
        shell("split", SPLIT_TEMPLATE, head_extra=REFRESH_STATE),
        state=SPLIT_STATE, group_dir=GROUP_DIR_NAME,
        message=request.args.get("msg", ""))


@app.route("/split/scan", methods=["POST"])
def split_scan():
    if SPLIT_STATE["status"] == "running":
        return redirect(url_for("split_page"))
    folder = request.form.get("folder", "").strip()
    if not folder or not Path(folder).is_dir():
        return redirect(url_for("split_page", msg="Esa carpeta no existe."))
    try:
        min_px = max(24, min(400, int(request.form.get("min_px", MIN_FACE_PX))))
    except ValueError:
        min_px = MIN_FACE_PX
    SPLIT_STATE.update(status="running", current=0, total=0, photo="", error="",
                       results=[], resumen={}, folder=folder, min_px=min_px, cancel=False)
    threading.Thread(target=_split_worker, args=(folder, min_px), daemon=True).start()
    return redirect(url_for("split_page"))


@app.route("/split/stop", methods=["POST"])
def split_stop():
    if SPLIT_STATE["status"] == "running":
        SPLIT_STATE["cancel"] = True
    return redirect(url_for("split_page"))


@app.route("/split/move", methods=["POST"])
def split_move():
    if SPLIT_STATE["status"] != "done" or not SPLIT_STATE["results"]:
        return redirect(url_for("split_page"))
    rels = request.form.getlist("rel")
    if not rels:
        return redirect(url_for("split_page", msg="No había nada tildado."))
    moved, errors = move_to_group(SPLIT_STATE["folder"], rels)
    done = set(moved)
    SPLIT_STATE["results"] = [g for g in SPLIT_STATE["results"] if g["rel"] not in done]
    msg = (f"{len(moved)} foto(s) grupales movidas a {GROUP_DIR_NAME}. "
           f"El importador ahora las saltea: aprendé de esta carpeta sin riesgo, "
           f"y procesá las grupales con el corrector.")
    if errors:
        msg += f" {len(errors)} no se pudieron mover (ej: {errors[0][0]})."
    return redirect(url_for("split_page", msg=msg))


@app.route("/split_thumb/<fn>")
def split_thumb(fn):
    return send_from_directory(SPLIT_THUMBS, fn)


@app.route("/split_file/<path:rel>")
def split_file(rel):
    return send_from_directory(SPLIT_STATE["folder"], rel)


@app.route("/cluster/discard", methods=["POST"])
def cluster_discard():
    """Descarta un grupo entero: son detecciones que NO son personas (tapizado,
    manchas, objetos que parecen caras). Se saca del análisis; no se organiza ni
    se vuelve a preguntar. No borra fotos del disco."""
    data = load_clusters()
    if not data:
        return {"ok": False}
    cid = str(request.form.get("cluster_id", ""))

    # Lista negra permanente: los vectores del grupo descartado se recuerdan
    # para que en los próximos análisis lo mismo se descarte solo.
    target = next((c for c in data["clusters"] if str(c["id"]) == cid), None)
    if target and ENCODINGS_NPY.is_file():
        try:
            enc = np.load(ENCODINGS_NPY)
            vecs = [enc[int(fid)] for fid in target["faces"] if int(fid) < len(enc)]
            if vecs:
                add_not_faces(vecs)
        except Exception:
            pass   # sin vectores no hay lista negra, pero el descarte sigue

    data["clusters"] = [c for c in data["clusters"] if str(c["id"]) != cid]
    labels = load_labels()
    if labels.pop(cid, None) is not None:
        save_labels(labels)
    record_verdict(cid, None)   # no era una persona: no cuenta para la calidad
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
    _record_suggestion_verdict(cluster_id, name)
    return {"ok": True, "name": name, "stats": stats_summary()}


def _record_suggestion_verdict(cluster_id, name):
    """Anota si la primera sugerencia del grupo acertó (métrica de calidad).
    Solo cuenta grupos que tenían sugerencia; quitar el nombre borra el
    veredicto y renombrar lo pisa."""
    data = load_clusters()
    if not data:
        return
    cluster = next((c for c in data["clusters"] if str(c["id"]) == str(cluster_id)), None)
    if not cluster or not cluster.get("suggestions"):
        return
    if not name:
        record_verdict(cluster_id, None)
    else:
        hit = name.strip().lower() == cluster["suggestions"][0].strip().lower()
        record_verdict(cluster_id, hit)


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
    # carpeta a mostrar: la de la búsqueda en curso, la de la última, o la del
    # análisis como sugerencia inicial
    folder = (DUPE_STATE.get("folder", "")
              or (data or {}).get("folder", "")
              or cfg.get("photos_dir", ""))
    return render_template_string(
        shell("dupes", DUPES_TEMPLATE, head_extra=REFRESH_STATE),
        folder=folder,
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
    folder = request.form.get("folder", "").strip()
    if not folder or not Path(folder).is_dir():
        return redirect(url_for("dupes_page", msg="Esa carpeta no existe."))
    folder_p = Path(folder)
    # Excluir la Biblioteca SOLO si está anidada dentro de la carpeta escaneada
    # (no re-deduplicar lo ya ordenado). Si escaneás DENTRO de la Biblioteca,
    # no se excluye nada — antes eso vaciaba todo y decía "no hay imágenes".
    library = get_library()
    exclude = None
    if library:
        libp = Path(library)
        if libp != folder_p and libp.is_relative_to(folder_p):
            exclude = libp
    DUPE_STATE.update(status="running", current=0, total=0, photo="", error="",
                      folder=folder)
    threading.Thread(target=_dupes_worker, args=(Path(folder), DEFAULT_THRESHOLD, exclude),
                     daemon=True).start()
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


@app.route("/dupes/clear", methods=["POST"])
def dupes_clear():
    """Vacía los resultados de duplicados: borra el índice y las miniaturas y
    vuelve la pantalla a blanco. No toca ninguna foto del usuario."""
    if DUPE_STATE["status"] == "running":
        return redirect(url_for("dupes_page"))
    DUPES_PATH.unlink(missing_ok=True)
    if DUPES_THUMBS.is_dir():
        import shutil
        shutil.rmtree(DUPES_THUMBS, ignore_errors=True)
    DUPE_STATE.update(status="idle", current=0, total=0, photo="", error="")
    return redirect(url_for("dupes_page", msg="Resultados de duplicados limpiados."))


@app.route("/dupe_photo/<path:rel>")
def dupe_photo(rel):
    """Sirve una foto original desde la carpeta de la última búsqueda de duplicados."""
    data = load_dupes()
    base = Path(data["folder"]) if data else Path(load_config().get("photos_dir", ""))
    return send_from_directory(base, rel)


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
        shell("rename", RENAME_TEMPLATE), f=form, example=_rename_example(form),
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
        shell("rename", RENAME_TEMPLATE), f=form, example=_rename_example(form),
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
