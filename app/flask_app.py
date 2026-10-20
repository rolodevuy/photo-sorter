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
import os
import sys
import threading
import time
import traceback
import unicodedata
import webbrowser
from pathlib import Path

import io
import zipfile

from flask import (Flask, Response, abort, redirect, render_template_string,
                   request, send_file, send_from_directory, stream_with_context,
                   url_for)

import numpy as np
from PIL import Image, ImageOps

from app import (CLUSTERS_PATH, CORRECT_THUMBS, DATA_DIR, DUPES_PATH, DUPES_THUMBS,
                 ENCODINGS_NPY, FACES_DIR, FIND_THUMBS, GALLERY_THUMBS, GROUP_DIR_NAME,
                 KNOWN_PATH, KNOWN_THUMBS, LABELS_PATH, PICK_THUMBS, VIDEO_EXTENSIONS)
from app.analyze import MIN_FACE_PX, run_analysis
from app.config import get_library, load_config, save_config, set_library
from app.corrector import apply_correction, scan_person_folder
from app.duplicates import (DEFAULT_THRESHOLD, DEFAULT_VIDEO_MAX_MB,
                            find_duplicates, resolve_duplicates)
from app.finder import (EPS_DEFAULT, EPS_MAX, EPS_MIN, find_known, find_person,
                        move_to_person)
from app.known import (add_faces, enroll_from_folder, forget_person,
                       import_folder_as_person, known_summary, library_people,
                       library_photo_counts, load_known, move_photo,
                       person_photos, save_face_crops, save_known, safe_name,
                       save_uploaded_photos, set_cover)
from app.notfaces import add_not_faces, clear_not_faces, count_not_faces
from app.nsfw import (NSFW_DIR_NAME, NSFW_THUMBS, SCORE_MIN, SCORE_MIN_HI,
                      SCORE_MIN_LO, move_to_nsfw, scan_nsfw)
from app import person_meta
from app import tags as tags_mod
from app import interrogator, vlm
from app import sexpos
from app import moldes as moldes_mod
from app import posts as posts_mod
from app import copetes as copetes_mod
from app import procesadas
from app.home_premium import build_context as prem_build_context
from app.home_premium import build_context_cached as prem_build_cached
from app.home_premium import HOME_TEMPLATE as PREMIUM_HOME
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


@app.after_request
def _sin_cache_html(resp):
    """Las PÁGINAS nunca se cachean: cada F5 trae la versión actual.

    Estuvimos días con 'no veo los cambios' y la mitad era el navegador
    sirviendo HTML viejo de su caché (ni el Ctrl+F5 salvaba siempre). Las
    miniaturas sí se cachean (van por send_from_directory con su mimetype de
    imagen); esto solo pisa el HTML."""
    if resp.mimetype == "text/html":
        resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/estatico/base.css")
def base_css():
    """El CSS común, servido aparte para que el navegador lo cachee.

    Antes iba embebido en cada página: 537 líneas repetidas en cada carga (el
    perfil pesaba 145 KB). Ahora se baja una vez. El ?v= del enlace lleva el
    hash del CSS, así que si lo editás cambia la URL y se ve al instante — no
    hay que acordarse de vaciar la caché."""
    return Response(BASE_CSS, mimetype="text/css",
                    headers={"Cache-Control": "public, max-age=31536000"})


@app.route("/debug/jserror", methods=["POST"])
def debug_jserror():
    """Los errores de JS del navegador llegan acá y se imprimen en esta
    consola — si no, quedan solo en la consola del navegador (F12) y nunca
    se ven en la terminal donde corre PhotoSorter.bat."""
    d = request.get_json(silent=True) or {}
    print(f"\n[JS ERROR] {d.get('url', '')}: {d.get('msg', '')}\n{d.get('extra', '')}\n", flush=True)
    return {"ok": True}


# Estado del análisis en curso (corre en un hilo aparte)
STATE = {"status": "idle", "current": 0, "total": 0, "photo": "", "error": ""}
# Estado de la búsqueda de duplicados (otro hilo)
DUPE_STATE = {"status": "idle", "current": 0, "total": 0, "photo": "", "error": ""}
# Estado de la importación de personas conocidas (otro hilo)
IMPORT_STATE = {"status": "idle", "current": 0, "total": 0, "photo": "", "error": ""}
# Estado del lote de posts (otro hilo): escribir el pie y/o los comentarios de
# TODAS las fotos de una persona. Es para dejarlo toda la noche: a ~90s el pie,
# 1500 fotos son más de 37 horas. Por eso guarda los tiempos REALES de cada foto
# y con eso calcula lo que falta, en vez de estimar con un número inventado.
POSTS_STATE = {"status": "idle", "current": 0, "total": 0, "photo": "", "error": "",
               "name": "", "que": "pie", "hechas": 0, "salteadas": 0, "fallaron": 0,
               "segundos": [], "empezo": 0.0, "stop": False, "ultimos": []}
# El etiquetador automático (otro lote, mismo patrón): selecciono fotos o dejo
# que corra sobre todas las que no tienen etiquetas propias todavía.
TAGS_STATE = {"status": "idle", "current": 0, "total": 0, "photo": "", "error": "",
              "name": "", "hechas": 0, "salteadas": 0, "fallaron": 0,
              "empezo": 0.0, "stop": False, "ultimos": []}
# Generador de borradores de moldes (otro hilo): recolecta combos de la
# biblioteca con umbral de frecuencia y le pide frases al modelo, de a una.
GEN_STATE = {"status": "idle", "current": 0, "total": 0, "combo": "", "error": "",
             "combos_nuevos": 0, "frases": 0, "empezo": 0.0, "stop": False, "ultimos": []}
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
    /* Paleta "GitHub oscuro" (mockup Organizar overlay). Se remapea sobre las
       variables que ya usaba todo el tema, así el cambio es global y de una. */
    --bg:#0d1117; --side:#0f141b; --panel:#161b22; --panel2:#21262d;
    --bd:#30363d; --tx:#e6edf3; --mut:#8b949e; --acc:#1f6feb; --acc2:#58a6ff;
    --grn:#3fb950; --red:#f85149; --amb:#f2cc60; --viol:#a371f7;
    --ov:rgba(13,17,23,.72);          /* fondo de overlays sobre imágenes */
    --line:#21262d;                   /* borde sutil (paneles internos del visor) */
  }
  * { box-sizing: border-box; }
  /* Inter si está instalada; si no, Segoe UI (casi igual). SIN Google Fonts:
     la app es offline y no quiero que dependa de internet por una fuente. */
  body { margin:0; background:var(--bg); color:var(--tx);
         font-family:"Inter","Segoe UI",system-ui,sans-serif; }
  a { color:var(--acc2); text-decoration:none; }
  a:hover { color:#79c0ff; text-decoration:underline; }
  h1 { margin:.2rem 0 .5rem; font-size:1.5rem; font-weight:650; }
  code { background:var(--bg); color:#c9d1d9; padding:1px 6px; border-radius:5px;
         border:1px solid var(--bd); font-size:.9em; }
  .hint-ico { display:inline-flex; align-items:center; justify-content:center;
              width:1.25rem; height:1.25rem; border-radius:50%; margin-left:.4rem;
              background:var(--panel2); color:var(--mut); border:1px solid var(--bd);
              font-size:.78rem; font-weight:700; font-style:normal; cursor:help;
              vertical-align:middle; user-select:none; }
  .hint-ico:hover { color:var(--acc2); border-color:var(--acc2); }
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
  .sidebar a.sec { font-weight:700; font-size:1rem; color:var(--tx); }
  .sidebar a.sec:not(.active) { color:var(--mut); }
  /* pestañas de la sección activa, arriba del contenido */
  .tabbar { display:flex; align-items:center; gap:.3rem; margin-bottom:1.1rem;
    flex-wrap:wrap; }
  .tabbar > .ic, .tabbar > span.tbl { font-size:.7rem; font-weight:700;
    letter-spacing:.09em; text-transform:uppercase; color:var(--mut);
    margin-right:.55rem; opacity:.85; }
  .tabbar a { display:inline-flex; align-items:center; gap:.5rem; padding:.42rem .9rem;
    border-radius:999px; color:var(--mut); font-size:.92rem; font-weight:600;
    text-decoration:none; background:transparent; border:1px solid transparent;
    transition:color .15s, background .15s, border-color .15s; }
  .tabbar a:hover { background:var(--panel2); color:var(--tx); }
  .tabbar a.active { background:var(--panel2); color:var(--tx); border-color:var(--bd);
    box-shadow:inset 3px 0 0 var(--acc); }
  .tabbar a .ic { width:1.2rem; text-align:center; font-size:1rem; }
  .tabbar .tbsearch { margin-left:auto; display:flex; align-items:center; gap:.4rem;
    background:var(--panel); border:1px solid var(--bd); border-radius:999px;
    padding:.32rem .9rem; }
  .tabbar .tbsearch .tic { color:var(--mut); font-size:.95rem; }
  .tabbar .tbsearch input { background:transparent; border:0; outline:0; color:var(--tx);
    font-size:.9rem; width:185px; }
  .tabbar .tbsearch input::placeholder { color:var(--mut); }
  .tabbar a.tbav { width:38px; height:38px; border-radius:50%; flex:none;
    background:var(--panel); border:1px solid var(--bd); color:var(--tx);
    font-size:1.05rem; display:flex; align-items:center; justify-content:center;
    text-decoration:none; margin-left:.35rem; transition:border-color .15s, transform .15s; }
  .tabbar a.tbav:hover { border-color:var(--acc2); transform:scale(1.06); }
  .main { flex:1; padding:2rem 2.6rem; max-width:1040px; min-width:0; }
  @media (max-width:720px) {
    .app { flex-direction:column; }
    .sidebar { width:auto; height:auto; position:static; display:flex;
               flex-wrap:wrap; gap:2px; }
    .sidebar .brand { width:100%; padding-bottom:.4rem; }
    .sidebar a.sec { width:auto; }
    .tabbar .tbsearch { width:100%; margin-left:0; }
    .tabbar .tbsearch input { flex:1; width:auto; }
    .main { padding:1.2rem; }
  }

  /* tarjetas / paneles */
  .panel, .card, .group { background:var(--panel); border:1px solid var(--bd);
    border-radius:12px; padding:1rem; margin-bottom:1rem; }

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
  button { padding:.5rem .9rem; font-size:.95rem; cursor:pointer; border-radius:8px;
    border:1px solid var(--bd); background:var(--panel2); color:var(--tx); }
  button:hover { background:#30363d; border-color:#484f58; }
  button:disabled { opacity:.45; cursor:default; }
  .primary { background:var(--acc); color:#fff; border:1px solid #388bfd; font-weight:600; }
  .primary:hover { background:#388bfd; border-color:#388bfd; }
  .secondary { background:var(--panel2); color:var(--tx); }
  .organize, .apply { background:#238636; color:#fff; border:1px solid #2ea043; font-weight:600;
    padding:.6rem 1.2rem; font-size:1.05rem; }
  .organize:hover, .apply:hover { background:#2ea043; }
  .danger { background:var(--red); color:#fff; border:0; }
  .dismiss { float:right; background:none; border:0; font-size:1rem; cursor:pointer;
    color:inherit; opacity:.55; padding:0 .2rem; }
  .dismiss:hover { opacity:1; }

  /* perilla de sensibilidad (NSFW): barra con relleno hasta el valor */
  input[type="range"].sens { -webkit-appearance:none; appearance:none; width:180px; height:6px;
    border-radius:999px; background:linear-gradient(90deg,var(--acc),var(--acc) var(--pct,50%),var(--bd) var(--pct,50%));
    outline:none; cursor:pointer; }
  input[type="range"].sens::-webkit-slider-thumb { -webkit-appearance:none; width:18px; height:18px;
    border-radius:50%; background:#fff; border:2px solid var(--acc); cursor:pointer; }
  input[type="range"].sens::-moz-range-thumb { width:18px; height:18px; border-radius:50%;
    background:#fff; border:2px solid var(--acc); cursor:pointer; }
  input[type="range"].sens::-moz-range-track { background:transparent; }

  /* avisos */
  .msg { background:rgba(63,185,80,.13); border:1px solid var(--grn); color:#aee0b9;
    padding:.75rem 1rem; border-radius:8px; margin-bottom:1rem; }
  .err { background:rgba(248,81,73,.13); border:1px solid var(--red); color:#f5b7b3;
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

  /* galería — tarjetas cuadradas estilo mockup */
  .grid { display:grid; grid-template-columns:repeat(auto-fill, minmax(210px, 1fr)); gap:16px; }
  .person { background:var(--panel); border:1px solid var(--bd); border-radius:12px;
    overflow:hidden; display:flex; flex-direction:column; }
  .person:hover { border-color:#484f58; }
  .person a { color:inherit; text-decoration:none; }
  /* la foto llena SIEMPRE un cuadrado, sin importar su forma original. El
     padding-top:100% arma un cuadrado (funciona en todos lados) y la imagen va
     absoluta adentro con object-fit:cover, así todas las cards quedan iguales. */
  .pthumb { display:block; position:relative; width:100%; padding-top:100%;
    overflow:hidden; cursor:pointer; }
  .person img { position:absolute; inset:0; width:100%; height:100%; object-fit:cover; display:block; }
  .noimg { position:absolute; inset:0; background:var(--panel2);
    display:flex; align-items:center; justify-content:center; font-size:3rem; color:var(--mut); }
  .badge-faces { position:absolute; top:8px; left:8px; background:var(--ov); border:1px solid var(--bd);
    border-radius:6px; padding:2px 8px; font-size:11px; color:#c9d1d9; backdrop-filter:blur(4px); }
  .badge-ok { position:absolute; top:8px; right:8px; width:24px; height:24px; border-radius:999px;
    background:var(--ov); border:1px solid var(--grn); color:var(--grn); display:flex; align-items:center;
    justify-content:center; font-size:13px; font-weight:700; backdrop-filter:blur(4px); }
  /* checkbox para etiquetar de a muchas: encima de la foto */
  .selbox { position:absolute; bottom:8px; left:8px; z-index:2; background:var(--ov);
    border:1px solid var(--bd); border-radius:6px; padding:.15rem .3rem; cursor:pointer; backdrop-filter:blur(4px); }
  .selbox input { width:1.1rem; height:1.1rem; cursor:pointer; accent-color:var(--acc); }
  .pbody { padding:12px; display:flex; flex-direction:column; gap:8px; }
  .pname { font-weight:700; font-size:15px; overflow:hidden; text-overflow:ellipsis;
    white-space:nowrap; }
  .pcount { color:var(--mut); font-size:.8rem; }
  .pfill { flex:1; }
  .forget-btn { width:100%; }
  .forget-btn:hover { background:#3d2226; border-color:var(--red); color:#ff7b72; }

  /* página de persona — card compacta de 2 columnas (mockup Panel Portada) */
  .pcard { display:grid; grid-template-columns:220px 1fr; background:var(--panel);
    border:1px solid var(--bd); border-radius:12px; overflow:hidden; margin-bottom:1rem; }
  .pcard-cover { background:var(--bg); border-right:1px solid var(--line); padding:16px;
    display:flex; flex-direction:column; gap:12px; }
  .cover-box { position:relative; width:100%; padding-top:100%; border-radius:10px;
    overflow:hidden; border:1px solid var(--bd); }
  .cover-box img, .cover-box .noimg { position:absolute; inset:0; width:100%; height:100%;
    object-fit:cover; }
  .cover-edit { position:absolute; bottom:8px; right:8px; z-index:2; width:32px; height:32px;
    border-radius:50%; background:var(--ov); backdrop-filter:blur(4px); font-size:14px; }
  .pcard-main { padding:16px 18px; position:relative; display:flex; flex-direction:column; gap:12px; }
  .iconbtn { width:38px; height:38px; border-radius:10px; display:flex; align-items:center;
    justify-content:center; font-size:17px; flex:none; }
  .pcfg-menu { position:fixed; z-index:20; width:340px; max-width:88vw;
    max-height:80vh; overflow-y:auto;
    background:var(--panel2); border:1px solid var(--bd); border-radius:12px;
    box-shadow:0 12px 32px rgba(0,0,0,.5); padding:12px; }
  .pcfg-sep { border:0; border-top:1px solid var(--bd); margin:10px 0; }
  .pcfg-item { display:block; width:100%; text-align:left; padding:9px 8px; border-radius:8px;
    color:var(--tx); background:none; border:0; font-size:.9rem; cursor:pointer; }
  .pcfg-item:hover { background:#262d3a; text-decoration:none; }
  .p-bio { color:var(--mut); font-size:.9rem; line-height:1.45; margin:.25rem 0 0;
    max-width:52ch; overflow-wrap:break-word; }
  @media (max-width:640px) { .pcard { grid-template-columns:1fr; }
    .pcard-cover { border-right:0; border-bottom:1px solid var(--line); } }
  /* carrusel de favoritos dentro de la card */
  .favwrap { display:flex; align-items:center; gap:6px; margin-top:.4rem; }
  .favnav { padding:.6rem .4rem; flex:none; }
  .favrow { flex:1; display:flex; gap:6px; overflow-x:auto; scroll-behavior:smooth;
    scrollbar-width:none; }
  .favrow img { width:150px; height:150px; flex:none; object-fit:cover; border-radius:10px;
    border:1px solid var(--line); cursor:pointer; }
  .favrow img:hover { border-color:var(--acc2); }
  .favplay { width:150px; height:150px; flex:none; border-radius:10px; border:1px dashed var(--bd);
    background:var(--bg); display:flex; flex-direction:column; align-items:center;
    justify-content:center; gap:3px; color:var(--mut); font-size:12px; line-height:1.2;
    text-align:center; }
  .favplay:hover { border-color:#484f58; color:var(--tx); text-decoration:none; }
  /* página playlist de favoritos */
  .favs-grid { display:grid; grid-template-columns:repeat(auto-fill, minmax(150px, 1fr));
    gap:12px; margin-top:1rem; }
  .favcell { position:relative; width:100%; padding-top:100%; border-radius:10px;
    overflow:hidden; border:1px solid var(--line); }
  .favcell:hover { border-color:var(--acc2); }
  .favcell img { position:absolute; inset:0; width:100%; height:100%; object-fit:cover;
    cursor:zoom-in; }
  .unfav { position:absolute; top:8px; right:8px; width:28px; height:28px;
    border-radius:999px; background:var(--ov); border:1px solid var(--bd); color:var(--amb);
    font-size:14px; cursor:pointer; backdrop-filter:blur(4px); padding:0; }
  .unfav:hover { border-color:var(--amb); }
  .favadd { border:1px dashed var(--bd); border-radius:10px; background:var(--bg);
    display:flex; flex-direction:column; align-items:center; justify-content:center;
    gap:4px; color:var(--mut); min-height:150px; font-size:12px; }
  .favadd:hover { border-color:#484f58; color:var(--tx); text-decoration:none; }
  /* insignia NSFW en la grilla limpia */
  .vid-badge { position:absolute; top:50%; left:50%; transform:translate(-50%,-50%);
    z-index:2; width:44px; height:44px; border-radius:50%; background:rgba(0,0,0,.55);
    color:#fff; display:flex; align-items:center; justify-content:center; font-size:18px;
    padding-left:3px; pointer-events:none; backdrop-filter:blur(2px); }
  .nsfw-badge { position:absolute; bottom:6px; left:6px; z-index:2; font-size:.85rem;
    background:var(--ov); border:1px solid var(--red); border-radius:6px; padding:1px 5px;
    backdrop-filter:blur(4px); }
  /* acciones de la foto en el visor */
  #lb.lb-post .pact { flex:0 0 auto; display:flex; align-items:center; gap:.5rem;
    padding:.5rem .9rem; border-top:1px solid var(--line); }
  /* galería del perfil: cuadrados uniformes (estilo mockup) — la foto llena un
     cuadrado y los controles van compactos abajo */
  .masonry { display:grid; grid-template-columns:repeat(auto-fill, minmax(160px, 1fr)); gap:12px; }
  @media (max-width:640px) { .masonry { grid-template-columns:repeat(auto-fill, minmax(120px, 1fr)); gap:8px; } }
  .masonry .shot { background:var(--panel); border:1px solid var(--bd); border-radius:10px;
    overflow:hidden; margin:0; text-align:center; display:flex; flex-direction:column; }
  .masonry .shot:hover { border-color:#484f58; }
  .shot-thumb { position:relative; width:100%; padding-top:100%; overflow:hidden; cursor:zoom-in; }
  .masonry .shot img { position:absolute; inset:0; width:100%; height:100%; object-fit:cover; display:block; }
  .shot-body { padding:.45rem; display:flex; flex-direction:column; gap:.3rem; }
  .shot .fn { font-size:.72rem; color:var(--mut); overflow:hidden;
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
  details.lote > summary { cursor:pointer; color:var(--acc); margin:.4rem 0; }
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

  /* revisión NSFW estilo mockup: tarjeta 3/4 con pill ✓% y etiquetas encima */
  .nsfw-grid { display:grid; grid-template-columns:repeat(auto-fill, minmax(180px, 1fr)); gap:14px; }
  .nsfw-grid .hit { width:auto; padding:0; overflow:hidden; background:var(--bg);
    border:1px solid var(--line); }
  .nsfw-grid .hit:has(input:checked) { border-color:var(--acc); }
  .hit-thumb { position:relative; width:100%; padding-top:133%; }
  .hit-thumb img { position:absolute; inset:0; width:100%; height:100%; max-width:none;
    max-height:none; object-fit:cover; cursor:zoom-in; border-radius:0; }
  .hit-pill { position:absolute; top:8px; left:8px; display:inline-flex; align-items:center;
    gap:6px; background:rgba(13,17,23,.78); border:1px solid var(--bd); border-radius:7px;
    padding:4px 9px; font-size:12px; font-weight:600; color:var(--tx); cursor:pointer;
    backdrop-filter:blur(4px); }
  .hit-pill input { accent-color:var(--acc); margin:0; cursor:pointer; }
  .hit-tags { position:absolute; bottom:8px; left:8px; right:8px; background:rgba(13,17,23,.78);
    border:1px solid var(--bd); border-radius:6px; padding:2px 8px; font-size:11px;
    color:#adbac7; backdrop-filter:blur(4px); overflow:hidden; text-overflow:ellipsis;
    white-space:nowrap; }

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
  /* dropdown de autocompletar etiquetas (tagAC, en BASE_JS): fixed, flota sobre
     lo que sea, para no romper el layout del input que lo dispara */
  .tagac { position:fixed; z-index:200; background:var(--panel2); border:1px solid var(--bd);
    border-radius:8px; padding:4px; box-shadow:0 8px 24px rgba(0,0,0,.5);
    max-height:220px; overflow-y:auto; display:none; }
  .tagac-item { padding:.3rem .6rem; border-radius:6px; font-size:.85rem; cursor:pointer;
    white-space:nowrap; }
  .tagac-item:hover, .tagac-item.on { background:var(--acc); color:#fff; }
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
  #lb { display:none; position:fixed; inset:0; background:rgba(1,4,9,.86); z-index:60;
    overflow:hidden; touch-action:none; backdrop-filter:blur(6px); }
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

  /* --- esqueletos de carga --- */
  /* La IA tarda entre 30s y 2 minutos por foto. Sin esto parece colgada: el
     usuario aprieta, no pasa nada visible, y no sabe si lo apretó. */
  @keyframes sk-brillo { 100% { background-position: -200% 0; } }
  .sk { border-radius:5px; background:linear-gradient(90deg,
          var(--panel2) 25%, var(--bd) 50%, var(--panel2) 75%);
    background-size:200% 100%; animation:sk-brillo 1.3s linear infinite; }
  .sk-linea { height:.7rem; margin:.45rem 0; }
  .sk-chip { display:inline-block; height:1.3rem; width:4.5rem; margin:.15rem .2rem; }
  /* respeta a quien pidió menos animación en el sistema */
  @media (prefers-reduced-motion: reduce) { .sk { animation:none; } }
  .trabajando { position:relative; opacity:.75; pointer-events:none; }

  /* --- visor estilo publicación (solo la galería de una persona) --- */
  /* foto a la izquierda, columna del post a la derecha */
  #lb.lb-post .post { position:absolute; inset:0; display:grid;
    grid-template-columns:minmax(0,1fr) 380px; }
  #lb.lb-post .stage { position:relative; inset:auto; overflow:hidden; }
  #lb.lb-post img { max-width:100%; max-height:100%; }
  /* botones flotantes sobre la foto (portada / personas) */
  .stage-fab { position:absolute; top:14px; z-index:4; box-shadow:0 2px 10px rgba(0,0,0,.5); }
  .stage-fab-l { left:14px; }
  .stage-fab-r { right:14px; }
  .stage-fab-bl { left:14px; top:auto; bottom:14px; }   /* "No es X" abajo-izquierda */
  .lb-tagprev { display:flex; flex-wrap:wrap; gap:.3rem; align-items:center; margin-bottom:.35rem; }
  .chip.mini { font-size:.72rem; padding:.12rem .45rem; }
  /* modal de "etiquetar persona" */
  .modal-ov { position:fixed; inset:0; z-index:60; background:var(--ov);
    display:flex; align-items:center; justify-content:center; overflow-y:auto; }
  /* overflow-y:auto: si la ventana queda chica, el overlay scrollea y ningún
     modal esconde contenido (la rueda no se queda "trabada") */
  /* el gestor de etiquetas se apila sobre el Centro de IA (que también es un
     .modal-ov): se abre desde el paso 2 sin obligarte a cerrar el creador */
  #lb-tag-modal { z-index:70; }
  .modal-box { background:var(--panel2); border:1px solid var(--bd); border-radius:12px;
    box-shadow:0 12px 32px rgba(0,0,0,.5); padding:14px; width:360px; max-width:92vw;
    max-height:80vh; display:flex; flex-direction:column; }
  .pmodal-list { overflow-y:auto; flex:1; }
  .pmodal-item { display:flex; align-items:center; gap:.6rem; padding:.5rem; border-radius:8px;
    cursor:pointer; }
  .pmodal-item:hover { background:#262d3a; }
  .pmodal-item img { width:34px; height:34px; border-radius:50%; object-fit:cover; background:var(--bg); flex:none; }
  .pmodal-item.added { opacity:.5; cursor:default; }
  #lb.lb-post .side { background:var(--bg); border-left:1px solid var(--line);
    display:flex; flex-direction:column; min-height:0; }
  #lb.lb-post .phead { display:flex; align-items:center; gap:.6rem; padding:.7rem .9rem;
    border-bottom:1px solid var(--line); flex:0 0 auto; }
  #lb.lb-post .av { width:38px; height:38px; border-radius:50%; object-fit:cover; }
  #lb.lb-post .phead .meta { font-size:.72rem; word-break:break-all; }
  /* el cuerpo scrollea; el perfil y el pie de la columna quedan fijos */
  #lb.lb-post .pbody { flex:1 1 auto; overflow-y:auto; padding:.8rem .9rem; min-height:0; }
  #lb.lb-post .favbtn { width:34px; height:34px; border-radius:8px; border:1px solid var(--bd);
    background:var(--panel2); font-size:1.15rem; cursor:pointer; color:var(--mut); line-height:1;
    flex:none; }
  #lb.lb-post .favbtn.on { color:var(--amb); border-color:var(--amb); background:rgba(242,204,96,.12); }
  /* la estrellita de la grilla: solo visible si la foto es favorita */
  .shot { position:relative; }
  .shot .favstar { position:absolute; top:.4rem; left:.4rem; z-index:2; display:none;
    font-size:1.1rem; filter:drop-shadow(0 1px 2px rgba(0,0,0,.6)); pointer-events:none; }
  .shot.is-fav .favstar { display:block; }
  /* checkbox para el etiquetador automático: arriba a la derecha, no choca con
     la estrella de favorito (arriba-izq) ni el badge NSFW (abajo-izq) */
  .shot-selbox { position:absolute; top:6px; right:6px; z-index:3; display:none;
    background:var(--ov); border:1px solid var(--bd); border-radius:6px;
    padding:.15rem .3rem; cursor:pointer; backdrop-filter:blur(4px); }
  .shot-selbox input { width:1.1rem; height:1.1rem; cursor:pointer; accent-color:var(--acc); }
  .shot.tagsel-on { outline:2px solid var(--acc); outline-offset:-2px; }
  #lb.lb-post .ppeople { margin-bottom:.8rem; padding-bottom:.6rem; border-bottom:1px solid var(--line); }
  /* pie y comentarios como sub-tarjetas con borde (mockup Visor Foto) */
  #lb.lb-post .pcard-sub { background:var(--panel); border:1px solid var(--panel2);
    border-radius:10px; padding:12px 14px; margin-bottom:1rem; }
  #lb.lb-post .pcard-sub > .meta:first-child { display:block; margin-bottom:.5rem; font-weight:600; }
  /* pie: texto a la izquierda + botones ✨/✏️ compactos a la derecha (mockup) */
  #lb.lb-post .caprow { display:flex; align-items:flex-start; gap:10px; }
  #lb.lb-post .capmain { flex:1; min-width:0; }
  #lb.lb-post .capfab { flex:none; width:34px; height:34px; min-width:0; padding:0;
    display:flex; align-items:center; justify-content:center; font-size:15px; }
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
  /* las sugerencias de IA pueden ser 40+: caja con scroll propio para que no
     empujen todo el panel */
  /* el modal de etiquetas: dos columnas — etiquetas a la izq, FOTO a la derecha
     (para chequear cada sugerencia contra lo que se ve) */
  .tagmodal { width:900px; max-width:96vw; }
  .tagm-cols { display:flex; gap:16px; flex:1; min-height:0; overflow:hidden; }
  .tagm-body { overflow-y:auto; flex:1 1 55%; min-width:0; min-height:0; padding-top:.2rem; }
  .tagm-photo { flex:1 1 45%; min-width:0; display:flex; align-items:flex-start;
    justify-content:center; }
  .tagm-photo img { max-width:100%; max-height:72vh; object-fit:contain; border-radius:10px;
    border:1px solid var(--bd); background:var(--bg); }
  @media (max-width:720px) {
    .tagm-cols { flex-direction:column-reverse; overflow-y:auto; }
    .tagm-photo img { max-height:38vh; }
  }
  .suglist { display:flex; flex-direction:column; gap:.35rem; margin-top:.5rem; }
  /* cada sugerencia: fila ANCHA con "+ agregar" grande y "✕ descartar" */
  .suglist .sugchip { display:flex; width:100%; align-items:stretch; background:var(--panel2);
    border:1px solid var(--bd); border-radius:8px; overflow:hidden; }
  .suglist .sugadd { flex:1; background:none; border:0; color:var(--tx); font-size:1rem;
    text-align:left; padding:.55rem .8rem; cursor:pointer; }
  .suglist .sugadd:hover { background:#262d3a; color:var(--acc2); }
  .suglist .sugx { background:none; border:0; border-left:1px solid var(--bd); color:var(--mut);
    cursor:pointer; font-size:.9rem; padding:0 .9rem; }
  .suglist .sugx:hover { background:#3d2226; color:var(--red); }
  .suglist .sugchip.sug-nueva { border-color:var(--amb); }
  .suglist .sugchip.sug-nueva .sugadd { color:var(--amb); }
  /* acto detectado por el clasificador dedicado: verde, va primero, más confiable */
  .suglist .sugchip.sug-acto { border-color:var(--grn); background:rgba(63,185,80,.1); }
  .suglist .sugchip.sug-acto .sugadd { color:var(--grn); font-weight:600; }
  /* sugerencia EDITABLE: marcador + input (para sacar "tipo de pelo:" y dejar "lacio") */
  .suglist .sugmark { align-self:center; padding:0 .3rem 0 .7rem; }
  .suglist .suginput { flex:1; background:none; border:0; color:var(--tx); font-size:1rem;
    padding:.55rem .4rem; min-width:0; }
  .suglist .suginput:focus { outline:none; background:#262d3a; }
  .suglist .sugchip.sug-acto .suginput { color:var(--grn); font-weight:600; }
  .suglist .sugchip.sug-nueva .suginput { color:var(--amb); }
  .suglist .sugadd2 { background:none; border:0; border-left:1px solid var(--bd);
    color:var(--acc2); cursor:pointer; font-size:1.05rem; padding:0 .8rem; }
  .suglist .sugadd2:hover { background:#262d3a; }
  /* preview de etiquetas editable (abajo de comentarios y paso 2 del Centro de IA) */
  #lb.lb-post .tagmini, #lb-ia-modal .tagmini { background:var(--bg); border:1px solid var(--bd); border-radius:6px;
    color:var(--tx); font-size:.78rem; padding:.12rem .4rem; width:6.5rem; }
  #lb.lb-post .chip .chiptxt, #lb-ia-modal .chip .chiptxt { cursor:text; }
  #lb.lb-post .lb-tagprev .chip.inherited, #lb-ia-modal .lb-tagprev .chip.inherited { opacity:.6; }
  /* "mostrar todas / menos" para las etiquetas propias cuando son muchas */
  #lb.lb-post .ownmore, #lb-ia-modal .ownmore { background:none; border:0; color:var(--acc2); cursor:pointer;
    font-size:.75rem; padding:.15rem .45rem; align-self:center; }
  #lb.lb-post .ownmore:hover, #lb-ia-modal .ownmore:hover { text-decoration:underline; }
  /* etiquetas al fondo, discretas: un link chico, no un botón grande, para que
     el pie y los comentarios se lleven toda la atención */
  #lb.lb-post .ptags-min { margin-top:.8rem; padding-top:.6rem;
    border-top:1px solid var(--line); display:flex; align-items:center; gap:.6rem;
    flex-wrap:wrap; }
  #lb.lb-post .linklike { background:none; border:0; color:var(--mut); cursor:pointer;
    font-size:.82rem; padding:0; }
  #lb.lb-post .linklike:hover { color:var(--acc2); text-decoration:underline; }
  /* Centro de IA: acordeón de pasos encadenados (Visión → Etiquetas → Pie →
     Comentarios → Copete). Cada paso es un <details>; el abierto muestra su
     cuerpo con el texto y el botoncito ✨ compacto. */
  .pia { flex:1 1 auto; min-height:0; display:flex; flex-direction:column; gap:.4rem;
    overflow-y:auto; padding:2px 8px 16px 4px; }
  /* crece hasta llenar el modal y scrollea adentro: los botones ✨ del paso
     abierto nunca quedan cortados por el borde del modal (el padding inferior
     garantiza aire para el último paso aunque el flex se quede corto) */
  .pia-step { background:var(--panel); border:1px solid var(--panel2);
    border-radius:10px; overflow:hidden; flex:0 0 auto; }
  /* flex:0 0 auto: los pasos no se aplastan para entrar (si faltara alto,
     scrollea el .pia con su barra; antes flex-shrink los encogía y los
     botones del último paso quedaban cortados sin forma de llegar) */
  .pia-step[open] { border-color:var(--bd); }
  .pia-step summary { display:flex; align-items:center;
    justify-content:space-between; gap:.6rem; padding:.55rem .8rem; cursor:pointer;
    font-size:.86rem; font-weight:600; color:var(--fg); list-style:none;
    user-select:none; }
  .pia-step summary::-webkit-details-marker { display:none; }
  .pia-step summary span.meta { font-weight:400; font-size:.75rem; }
  .pia-step summary::before { content:'▸'; color:var(--mut);
    font-size:.7rem; transition:transform .15s; }
  .pia-step[open] summary::before { transform:rotate(90deg); }
  .pia-body { padding:.5rem .8rem .7rem; border-top:1px solid var(--line);
    display:flex; flex-direction:column; gap:.45rem; min-width:0; }
  .pia-body p { margin:0; line-height:1.5; white-space:pre-wrap;
    font-size:.88rem; }
  .pia-actions { display:flex; align-items:center; gap:.45rem;
    flex-wrap:wrap; }
  .pia-tags { border-top:0; }
  /* el botón que abre el Centro de IA, al pie de la columna del post */
  .cat-button { width:100%; padding:.6rem; margin-top:.4rem; font-weight:600; }
  /* modal propio del Centro de IA */
  .iamodal { width:min(720px,96vw); overflow:hidden; }
  .iamodal .capfab { flex:none; width:34px; height:34px; min-width:0; padding:0;
    display:flex; align-items:center; justify-content:center; font-size:15px; }
  .iamodal .capin { width:100%; margin-top:.4rem; min-height:5rem; resize:vertical;
    background:var(--bg); color:var(--fg); border:1px solid var(--line);
    border-radius:6px; padding:.4rem; font:inherit; font-size:.85rem; }
  #lb-ia-modal .com { padding:.3rem 0; font-size:.88rem; line-height:1.4;
    border-bottom:1px solid var(--line); }
  #lb-ia-modal .com b { color:var(--fg); }
  #lb-ia-modal .empty { color:var(--dim); font-style:italic; }
  #lb-ia-modal .linklike { background:none; border:0; color:var(--mut);
    cursor:pointer; font-size:.82rem; padding:0; }
  #lb-ia-modal .linklike:hover { color:var(--acc2); text-decoration:underline; }
  /* sugerencia nueva/dudosa (no está en tu vocabulario): marcada, no borrada */
  #lb.lb-post .sugchip.sug-nueva { border-color:var(--amb); }
  #lb.lb-post .sugchip.sug-nueva .sugadd { color:var(--amb); }
  #lb.lb-post .pfoot { flex:0 0 auto; display:flex; justify-content:space-between;
    align-items:center; gap:.5rem; padding:.5rem .9rem; border-top:1px solid var(--line); }
  #lb.lb-post .pfoot select { font-size:.75rem; max-width:230px; }
  #lb.lb-post .sm { padding:.15rem .5rem; font-size:.78rem; }
  #lb.lb-post .sm.on { border-color:var(--accent); color:var(--accent); }
  /* los botones que disparan a la IA: grandes, porque son los que hay que
     encontrar y porque hay que ver clarito que quedaron apretados */
  #lb.lb-post .act, #lb-ia-modal .act, #lb-tag-modal .act { padding:.4rem .8rem; font-weight:600; min-width:11rem; }
  #lb.lb-post .act.trabajando, #lb-ia-modal .act.trabajando, #lb-tag-modal .act.trabajando { opacity:1; border-color:var(--accent); color:var(--accent);
    font-variant-numeric:tabular-nums; }
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

# Versión del CSS = hash de su contenido. Va en el ?v= del <link>: si editás el
# CSS cambia la URL y el navegador lo vuelve a pedir, sin tener que acordarse de
# vaciar la caché (y sin perder el cacheo del resto del tiempo).
CSS_VER = hashlib.sha1(BASE_CSS.encode("utf-8")).hexdigest()[:10]

# Cuántas fotos por página en la galería de una persona. Sin paginar, 1500fotos
# son 1500 <select> con toda la gente adentro y el navegador se arrastra.
PER_OPTIONS = [10, 20, 50, 100]
PER_DEFAULT = 50

# Secciones del menú: el sidebar muestra SOLO los títulos; cada uno abre su
# sección y las opciones aparecen como pestañas arriba del contenido.
# (título, ícono de sección, [(ícono, etiqueta, endpoint Flask, clave activa)])
NAV_SECTIONS = [
    ("Inicio", "🏠", [
        ("🏠", "Inicio", "inicio", "inicio"),
    ]),
    ("Explorar", "🔍", [
        ("☻", "Galería", "known_page", "known"),
        ("▦", "Rostros", "home", "home"),
        ("⌕", "Persona", "finder_page", "finder"),
    ]),
    ("Limpieza", "🧹", [
        ("⧉", "Duplicados", "dupes_page", "dupes"),
        ("✎", "Carpeta", "rename_page", "rename"),
        ("⚠", "NSFW", "nsfw_page", "nsfw"),
        ("⚇", "Separar caras", "split_page", "split"),
    ]),
    ("Contenido", "🖊", [
        ("✍", "Pies", "moldes_page", "moldes"),
        ("✎", "Frases", "phrases_page", "phrases"),
        ("🧪", "Probador", "probador", "probador"),
        ("🔁", "Conversiones", "conversiones_page", "conversiones"),
    ]),
    ("Sistema", "⚙", [
        ("⚙", "Ajustes", "settings_page", "settings"),
    ]),
]

# clave de página activa -> (título de sección, clave de pestaña activa)
_ACTIVE_TO_SECTION = {key: (title, key) for title, _ic, tabs in NAV_SECTIONS
                      for _i, _l, _ep, key in tabs}


def _seccion_de(active):
    return _ACTIVE_TO_SECTION.get(active)


def _nav_html(active):
    """Barra lateral: solo los TÍTULOS de sección. Entrás a la sección y las
    opciones son las pestañas (ver _tabbar_html)."""
    links = []
    for title, icon, tabs in NAV_SECTIONS:
        keys = {k for _i, _l, _ep, k in tabs}
        cls = "active" if active in keys else ""
        href = url_for(tabs[0][2])
        links.append(f'<a class="sec {cls}" href="{href}">'
                     f'<span class="ic">{icon}</span>{title}</a>')
    return ('<nav class="sidebar"><div class="brand">photo-sorter'
            '<small>reconocimiento local</small></div>' + "".join(links) + "</nav>")


def _tab_search_html():
    """Búsqueda rápida al final de la fila de pestañas (va en todas las
    secciones; en Inicio, que no tiene pestañas, queda sola a la derecha)."""
    return ('<form class="tbsearch" action="' + url_for("finder_page") + '" method="get">'
            '<span class="tic">🔍</span>'
            '<input name="q" placeholder="Buscar foto o persona" aria-label="Buscar">'
            '</form>')


def _tabbar_html(active):
    """Pestañas de la sección activa (arriba del contenido de cada página).
    Con una sola pestaña (Inicio, Sistema) no se dibujan pills: sería una
    pestaña suelta sin sentido; queda solo el buscador al final."""
    info = _seccion_de(active)
    if not info:
        return ""
    title, sub = info
    tabs = next(t for t in NAV_SECTIONS if t[0] == title)[2]
    parts = ['<div class="tabbar">']
    if len(tabs) > 1:
        parts.append(f'<span class="tbl">{title}</span>')
        for tab_icon, label, endpoint, key in tabs:
            if not key:
                continue
            cls = "active" if key == sub else ""
            parts.append(f'<a class="{cls}" href="{url_for(endpoint)}">'
                         f'<span class="ic">{tab_icon}</span>{label}</a>')
    parts.append(_tab_search_html())
    parts.append('<a class="tbav" href="' + url_for("settings_page") + '" title="Perfil">👤</a>')
    parts.append('</div>')
    return "".join(parts)


# Autocompletar de etiquetas, compartido por TODAS las páginas (va en el shell,
# no en cada template): lee las opciones del <datalist id="tagsug"> que cada
# página ya rellena server-side, y arma un dropdown propio que tabea rápido
# (el <datalist> nativo del navegador es lento/feo y no da flechitas ni Tab).
# Va en el <head>, ANTES que cualquier otro script de la página: si el
# listener se registra al final (como estaba, en BASE_JS), se pierde
# cualquier error que pase durante la carga inicial de scripts anteriores
# — que es justo donde suelen estar los bugs que rompen todo el resto.
EARLY_JS = """
<script>
// manda los errores de JS del navegador a la consola donde corre la app
// (si no, quedan solo en la consola del navegador y no se ven acá)
(function() {
  function report(msg, extra) {
    try {
      fetch('/debug/jserror', {method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({msg: String(msg), extra: extra || '', url: location.pathname})});
    } catch (e) {}
  }
  addEventListener('error', e => {
    report(e.message, (e.filename || '') + ':' + e.lineno + ':' + e.colno + '\\n' + (e.error && e.error.stack || ''));
  });
  addEventListener('unhandledrejection', e => {
    report('promesa sin atrapar: ' + (e.reason && e.reason.message || e.reason), e.reason && e.reason.stack || '');
  });
})();
</script>
"""

BASE_JS = """
<script>
/* Helpers compartidos. Estaban repetidos: `norm` y `esc` definidos dos veces, y
   el header de fetch copiado 22 veces. Acá van una sola vez, para todas las
   páginas. Los templates que ya los definen los pisan sin romper nada. */
function norm(s) {          // minúsculas y sin tildes, para comparar sin exactitud
  return (s || '').toLowerCase().normalize('NFD').replace(/[\\u0300-\\u036f]/g, '');
}
function esc(s) {           // texto seguro para meter en innerHTML
  return String(s).replace(/[<>&]/g, c => ({'<':'&lt;','>':'&gt;','&':'&amp;'}[c]));
}
function postForm(url, body) {   // POST de formulario -> json (el 90% de los fetch)
  return fetch(url, {method: 'POST',
    headers: {'Content-Type': 'application/x-www-form-urlencoded'},
    body: new URLSearchParams(body)}).then(r => r.json());
}
function tagAC(input, opts) {
  if (!input) return;
  opts = opts || {};
  const multi = !!opts.multi;      // true = el input puede llevar "tag1, tag2" (bulk)
  const commit = opts.commit;      // (valor) => agregar esa etiqueta (modo simple)
  const box = document.createElement('div');
  box.className = 'tagac';
  document.body.appendChild(box);
  let items = [], active = -1;
  const norm = s => (s || '').toLowerCase().normalize('NFD').replace(/[\\u0300-\\u036f]/g, '');

  function place() {
    const r = input.getBoundingClientRect();
    box.style.left = r.left + 'px';
    box.style.top = (r.bottom + 4) + 'px';
    box.style.width = Math.max(170, r.width) + 'px';
  }
  function hide() { box.style.display = 'none'; input.dataset.acOpen = ''; active = -1; }
  function typed() {
    if (!multi) return input.value.trim();
    const parts = input.value.split(',');
    return parts[parts.length - 1].trim();
  }
  function apply(tag) {
    if (multi) {
      const parts = input.value.split(',');
      parts[parts.length - 1] = ' ' + tag;
      input.value = parts.map(p => p.trim()).filter(Boolean).join(', ') + ', ';
    } else {
      if (commit) commit(tag);
      input.value = '';
    }
    input.focus();
    hide();
  }
  function highlight() {
    [...box.children].forEach((c, i) => c.classList.toggle('on', i === active));
  }
  function render() {
    const all = [...document.querySelectorAll('#tagsug option')].map(o => o.value);
    const q = norm(typed());
    if (!q) { hide(); return; }
    items = all.filter(t => norm(t).includes(q) && norm(t) !== q).slice(0, 8);
    if (!items.length) { hide(); return; }
    box.innerHTML = '';
    items.forEach(t => {
      const d = document.createElement('div');
      d.className = 'tagac-item';
      d.textContent = t;
      d.onmousedown = e => { e.preventDefault(); apply(t); };
      box.appendChild(d);
    });
    active = -1;
    place();
    box.style.display = 'block';
    input.dataset.acOpen = '1';
  }
  input.addEventListener('input', render);
  // TODO el manejo de Enter vive ACÁ, en un solo lugar. Si además hubiera un
  // onkeydown propio en el <input> (como el de ptag-input, que agrega el texto
  // tipeado), los dos se disparan en la misma tecla y en modo simple duplicaba
  // la etiqueta (la tipeada Y la resaltada). Por eso single-mode NO lleva
  // onkeydown aparte: todo el Enter pasa por acá.
  input.addEventListener('keydown', e => {
    const open = box.style.display !== 'none';
    if (open && e.key === 'ArrowDown') { e.preventDefault(); active = Math.min(active + 1, items.length - 1); highlight(); return; }
    if (open && e.key === 'ArrowUp') { e.preventDefault(); active = Math.max(active - 1, 0); highlight(); return; }
    if (open && (e.key === 'Tab' || (e.key === 'Enter' && active >= 0))) { e.preventDefault(); apply(items[active]); return; }
    if (e.key === 'Enter' && !multi) {
      // sin sugerencia resaltada: confirma lo tipeado tal cual (modo simple)
      e.preventDefault();
      const v = input.value.trim();
      if (v && commit) commit(v);
      input.value = '';
      hide();
      return;
    }
    if (e.key === 'Escape') { e.stopPropagation(); hide(); }
  });
  input.addEventListener('blur', () => setTimeout(hide, 150));
  window.addEventListener('scroll', () => { if (box.style.display !== 'none') place(); }, true);
}
</script>
"""


def shell(active, inner, head_extra=""):
    """Envuelve el contenido `inner` (HTML+Jinja de la página) con el <head>
    del tema oscuro y la barra lateral. Todo se renderiza junto por Jinja."""
    return (
        '<!doctype html><html lang="es"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        "<title>photo-sorter</title>" + EARLY_JS + BASE_JS + head_extra +
        # el CSS va por su propia URL para que el navegador lo cachee; el ?v= es
        # el hash del contenido, así editarlo se ve al instante
        '<link rel="stylesheet" href="/estatico/base.css?v=' + CSS_VER + '">'
        "</head><body>"
        '<div class="app">' + _nav_html(active) + '<main class="main">'
        + _tabbar_html(active) + inner +
        "</main></div>" + UNDO_FAB + CONFIRM_JS + "</body></html>")


# Botón flotante "↩ Deshacer": aparece si el servidor tiene algo para revertir.
# Va en TODAS las páginas (dentro del shell). Chequea al cargar y expone
# window.refrescarUndo() para que las acciones AJAX lo muestren en el momento.
# Confirmaciones por data-confirm en vez de onclick/onsubmit="confirm('…')".
# Un nombre o etiqueta con apóstrofo ("Iara D'Angelo") partía el string de JS y
# rompía el atributo entero, y con él el HTML que seguía. En un atributo normal
# el escapado lo hace el navegador y no hay nada que romper.
CONFIRM_JS = """
<script>
document.addEventListener('submit', e => {
  const m = e.target.getAttribute && e.target.getAttribute('data-confirm');
  if (m && !confirm(m)) e.preventDefault();
}, true);
document.addEventListener('click', e => {
  const el = e.target.closest && e.target.closest('[data-confirm]');
  if (el && el.tagName === 'BUTTON' && !confirm(el.getAttribute('data-confirm'))) {
    e.preventDefault(); e.stopPropagation();
  }
}, true);
</script>"""


UNDO_FAB = """
<div id="undo-fab" style="display:none; position:fixed; left:50%; transform:translateX(-50%); bottom:1rem; z-index:18">
  <button onclick="hacerUndo()" title="Deshacer la última acción"
    style="background:var(--pan); color:var(--tx); border:1px solid var(--bd);
    border-radius:999px; padding:.55rem 1rem; cursor:pointer; font-size:.9rem;
    box-shadow:0 6px 20px rgba(0,0,0,.5)">↩ Deshacer <span id="undo-desc" style="color:var(--mut)"></span></button>
</div>
<script>
(function() {
  function refrescar() {
    fetch('/undo/estado').then(r => r.json()).then(d => {
      const fab = document.getElementById('undo-fab');
      if (d && d.puede) { document.getElementById('undo-desc').textContent = d.desc || ''; fab.style.display = ''; }
      else fab.style.display = 'none';
    }).catch(() => {});
  }
  window.refrescarUndo = refrescar;
  window.hacerUndo = function() {
    document.getElementById('undo-fab').style.display = 'none';
    fetch('/undo', {method: 'POST'}).then(r => r.json()).then(() => location.reload())
      .catch(() => location.reload());
  };
  window.addEventListener('DOMContentLoaded', refrescar);
})();
</script>"""


# Metas de auto-refresco (van al <head> vía head_extra) según el estado en curso
REFRESH_STATE = '{% if state.status == "running" %}<meta http-equiv="refresh" content="2">{% endif %}'
REFRESH_IMPORT = '{% if imp.status == "running" %}<meta http-equiv="refresh" content="2">{% endif %}'

HOME_BODY = """
<h1>Ordenar por rostro <span class="hint-ico" title="Al organizar, tus fotos se mueven al destino como nombre_0000.jpg, separadas por persona (desaparecen del origen). Las fotos sin nombre quedan donde estaban. Todo corre en esta máquina.">ⓘ</span></h1>

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
<h1>Buscar duplicados <span class="hint-ico" title="Encuentra fotos repetidas: exactas (mismo archivo con otro nombre) y parecidas (la misma imagen reescalada o recomprimida). En cada grupo se marca cuál conviene conservar (mayor resolución). Al resolver, las sobrantes se mueven a una carpeta _duplicados dentro de la carpeta elegida — no se borran, las revisás vos.">ⓘ</span></h1>

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
          <div>{% if f.media == 'video' %}🎬 {% endif %}{{ f.w }}×{{ f.h }} ·
            {% if f.size >= 1048576 %}{{ (f.size/1048576)|round(1) }} MB{% else %}{{ (f.size/1024)|round|int }} KB{% endif %}</div>
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
<h1>Corregir carpeta (renombrar) <span class="hint-ico" title="Renombra en masa las imágenes y videos de una carpeta con el patrón que elijas: una palabra, un separador y un número que se va sumando solo (0000, 0001…). Solo cambia nombres (no toca el contenido). Trabaja solo en el primer nivel de la carpeta (no entra en subcarpetas).">ⓘ</span></h1>

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
      <button class="apply" data-confirm="Se renombrarán {{ total }} archivo(s). ¿Continuar?">
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


_PREMIUM_WIDE = """<style>
  .main { max-width: 1420px !important; }
</style>"""


@app.route("/inicio")
def inicio():
    """Portada premium: biblioteca fotográfica editorial (foto del día,
    protagonistas, recién, en-este-día, colecciones, descubrimientos)."""
    cfg = load_config()
    library = get_library(cfg)
    ctx = prem_build_cached(library)
    return render_template_string(
        shell("inicio", PREMIUM_HOME, head_extra=_PREMIUM_WIDE), **ctx)


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
<div class="row" style="justify-content:space-between;align-items:center;margin-bottom:.2rem">
  <h1 style="margin:0">Galería</h1>
  {% if known_dir %}
  <button class="secondary" onclick="cfgAbrir()" title="Configuración de la galería">⚙ Configuración</button>
  {% endif %}
</div>
{% if message %}<div class="msg">{{ message }}</div>{% endif %}

{% if not known_dir %}
<div class="err">Todavía no elegiste la <b>Biblioteca</b>. Andá a
  <a href="{{ url_for('settings_page') }}">⚙ Ajustes</a> y elegí la carpeta donde viven tus
  personas (una subcarpeta por cada una).</div>
{% else %}
{# lo de configurar la galería vive acá adentro: se usa de vez en cuando y antes
   empujaba las personas para abajo en cada visita #}
<div id="gal-cfg-modal" class="modal-ov" style="display:none" onclick="if(event.target===this) cfgCerrar()">
  <div class="modal-box" style="width:620px">
    <div class="row" style="justify-content:space-between;margin:0 0 .6rem">
      <b style="font-size:1.05rem">⚙ Configuración de la galería</b>
      <button class="secondary sm" onclick="cfgCerrar()" title="Cerrar">✕</button>
    </div>
    <div style="overflow-y:auto">
      <div class="row" style="margin-top:0">
        <span class="meta">📚 Biblioteca: <code>{{ known_dir }}</code></span>
        <a href="{{ url_for('settings_page') }}" class="meta">cambiar en Ajustes →</a>
      </div>

      <form method="post" action="{{ url_for('known_import') }}" style="margin-top:.9rem">
        <b>📥 Aprender caras <span class="hint-ico" title="Aprende de cada subcarpeta y suma personas nuevas sin reetiquetar. Se puede repetir: a los ya conocidos solo les suma ejemplos. Saltea _grupo y _revisar, así solo aprende de fotos donde la persona está sola.">ⓘ</span></b>
        <div class="row" style="margin:0">
          <button class="primary" {% if imp.status == 'running' %}disabled{% endif %}>
            📥 Aprender las caras de la Biblioteca
          </button>
          {% if people %}
          <button formaction="{{ url_for('known_clear') }}"
                  onclick="return confirm('¿Vaciar la base de personas conocidas? (no toca las fotos)')">Vaciar base</button>
          {% endif %}
        </div>
      </form>

      <hr style="border:0;border-top:1px solid var(--line);margin:1rem 0">

      <form method="post" action="{{ url_for('known_import_folder') }}">
        <b>➕ Agregar una carpeta como persona <span class="hint-ico" title="Elegís una carpeta y un nombre y todas las fotos de adentro se copian como esa persona, sin analizar caras. Para cuando ya sabés de quién son (ej: una carpeta «Alfredo» → todas son de Alfredo). Las originales no se tocan (se copian).">ⓘ</span></b>
        <div class="row" style="margin:0">
          <input class="path" type="text" name="folder" id="imp-folder" placeholder="Carpeta con las fotos…">
          <button type="button" onclick="browseImport()">📂 Elegir carpeta…</button>
        </div>
        <div class="row" style="margin-top:.45rem">
          <label>Nombre:</label>
          <input class="path" type="text" name="name" id="imp-name"
                 placeholder="ej: Alfredo (si lo dejás vacío, usa el nombre de la carpeta)">
          <button class="primary">➕ Agregar como persona</button>
        </div>
      </form>
    </div>
  </div>
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

function cfgAbrir() {      // ⚙ al lado del título: aprender caras + agregar carpeta
  const m = document.getElementById('gal-cfg-modal');
  if (m) m.style.display = 'flex';   // flex, no block: .modal-ov centra con flexbox
}
function cfgCerrar() {
  const m = document.getElementById('gal-cfg-modal');
  if (m) m.style.display = 'none';
}
document.addEventListener('keydown', e => {   // Escape cierra, como los otros modales
  if (e.key !== 'Escape') return;
  const m = document.getElementById('gal-cfg-modal');
  if (m && m.style.display !== 'none') cfgCerrar();
});

function browseImport() {   // carpeta a importar; autocompleta el nombre si está vacío
  fetch('{{ url_for('browse') }}').then(r => r.json()).then(d => {
    if (!d.path) return;
    document.getElementById('imp-folder').value = d.path;
    const nm = document.getElementById('imp-name');
    const base = d.path.replace(/\\\\/g, '/').split('/').filter(Boolean).pop() || '';
    if (!nm.value.trim()) nm.value = base;
  });
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
  // qué tokens son etiquetas reales (para arrastrar el filtro al perfil). OJO:
  // tiene que ser una etiqueta EXACTA, no subcadena — si no, buscar "ai" (por
  // aisha) matchea "bailando" y arrastra ?tags=ai al perfil, que filtra sus
  // fotos por "ai" y no muestra ninguna.
  const tagToks = tokens.filter(t => ALL_TAGS.some(x => norm(x) === t));
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
  const v = document.getElementById('selvis');
  if (v) v.textContent = visibles;
}

// --- etiquetar de a muchas ---
// El caso: "estas 60 son hombres". Entrar una por una no es una opción.
const BULK_URL = "{{ url_for('tags_person_bulk') }}";
let selOn = false;

function visibles() {
  return [...document.querySelectorAll('.person[data-name]')]
    .filter(c => c.style.display !== 'none');
}
function toggleSel() {
  selOn = !selOn;
  document.getElementById('selbar').style.display = selOn ? '' : 'none';
  document.getElementById('selbtn').textContent = selOn ? '✕ Salir de etiquetar' : '☑ Etiquetar varias';
  document.querySelectorAll('.selbox').forEach(b => b.style.display = selOn ? '' : 'none');
  if (!selOn) selAll(false);
  filterPeople();
}
function selAll(on) {
  // solo las que se ven: si filtraste "gonza", no te lleva puestas las demás
  const cards = on ? visibles() : [...document.querySelectorAll('.person[data-name]')];
  cards.forEach(c => { const chk = c.querySelector('.selchk'); if (chk) chk.checked = on; });
  selCount();
}
function selCount() {
  const n = document.querySelectorAll('.selchk:checked').length;
  document.getElementById('selcount').textContent = n;
  return n;
}
function bulk(accion) {
  const st = document.getElementById('bulkst');
  const nombres = [...document.querySelectorAll('.selchk:checked')].map(c => c.value);
  const tags = document.getElementById('bulktag').value.trim();
  if (!nombres.length) { st.textContent = ' elegí al menos una persona'; return; }
  if (!tags) { st.textContent = ' escribí la etiqueta'; return; }
  st.textContent = ' guardando…';
  fetch(BULK_URL, {method:'POST', headers:{'Content-Type':'application/x-www-form-urlencoded'},
    body: new URLSearchParams({names: nombres.join('\\n'), tags: tags, action: accion})})
    .then(r => r.json()).then(d => {
      if (!d.ok) { st.textContent = ' ' + (d.error || 'error'); return; }
      // repintar las tarjetas tocadas sin recargar (con 500 personas importa)
      for (const [nombre, sus] of Object.entries(d.tags)) {
        const card = document.querySelector(`.person[data-name="${CSS.escape(nombre)}"]`);
        if (!card) continue;
        // solo actualizamos el dato para el filtro; la tarjeta no muestra tags
        // (como en el mockup). Las etiquetas se ven/editan dentro de la persona.
        card.dataset.tags = sus.join(' ');
      }
      const verbo = accion === 'add' ? 'agregada a' : 'quitada de';
      st.textContent = ` ✓ ${verbo} ${d.n} persona(s)`;
      setTimeout(() => st.textContent = '', 2500);
    }).catch(e => st.textContent = ' ' + e);
}
</script>

{% if people %}
<div class="row" style="margin-bottom:.75rem">
  <input type="text" id="pfilter" class="path" placeholder="🔍 Buscar por nombre o etiqueta (coma para varias)…"
         oninput="filterPeople()" autofocus>
  <span class="meta"><span id="pcount">{{ people|length }}</span> persona(s)</span>
  <button class="secondary" onclick="toggleSel()" id="selbtn">☑ Etiquetar varias</button>
</div>

{# Etiquetar de a muchas: "todas estas son hombres" sin entrar una por una #}
<div class="panel" id="selbar" style="display:none">
  <div class="row">
    <button class="secondary" onclick="selAll(true)">Seleccionar los {{ '' }}<span id="selvis">0</span> que se ven</button>
    <button class="secondary" onclick="selAll(false)">Ninguna</button>
    <span class="meta"><b id="selcount">0</b> seleccionada(s)</span>
  </div>
  <div class="row" style="margin-top:.4rem">
    <input class="path" id="bulktag" placeholder="etiqueta(s), coma para varias… ej: hombre"
           onkeydown="if(event.key==='Enter'){event.preventDefault();bulk('add')}">
    <button class="primary" onclick="bulk('add')">＋ Agregar a las seleccionadas</button>
    <button class="secondary" onclick="bulk('remove')" title="Sacarles esta etiqueta">－ Quitar</button>
    <span class="meta" id="bulkst"></span>
    <span class="hint-ico" title="Agregar suma la etiqueta: no pisa las que ya tengan. El filtro de arriba funciona igual, así que podés buscar y seleccionar solo los que quedaron a la vista.">ⓘ</span>
  </div>
</div>
<datalist id="tagsug">{% for t in all_tags %}<option value="{{ t }}">{% endfor %}</datalist>
<div class="grid">
  {% for p in people %}
  <div class="person" id="p{{ loop.index0 }}" data-name="{{ p.name }}"
       data-tags="{{ p.tags|join(' ') }}" data-href="{{ url_for('person_page', name=p.name) }}">
    <a class="pthumb" href="{{ url_for('person_page', name=p.name) }}" title="Ver las fotos de {{ p.name }}">
      {% if p.thumb %}
        <img src="{{ url_for('known_thumb', name=p.name) }}" alt="{{ p.name }}">
      {% else %}
        <div class="noimg">👤</div>
      {% endif %}
      <span class="badge-faces">{{ p.fotos }} fotos</span>
      {% if p.corrected %}<span class="badge-ok" title="Corregido {{ p.corrected }}">✓</span>{% endif %}
      <label class="selbox" style="display:none" title="Seleccionar a {{ p.name }}">
        <input type="checkbox" class="selchk" value="{{ p.name }}" onchange="selCount()">
      </label>
    </a>
    <div class="pbody">
      <div class="pname">{{ p.name }}</div>
      <div class="pfill"></div>
      <form method="post" action="{{ url_for('known_forget') }}"
            data-confirm="¿Olvidar a {{ p.name }}? Se dejará de reconocer sola.">
        <input type="hidden" name="name" value="{{ p.name }}">
        <button class="forget-btn">🗑 Olvidar</button>
      </form>
    </div>
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
    people = known_summary(get_library())
    tagsets = tags_mod.people_tagsets()
    cuentas = library_photo_counts(get_library())
    for p in people:   # "corregido hace X" + etiquetas (suyas + de sus fotos)
        p["corrected"] = person_meta.humanize(person_meta.get(p["name"])["corrected"])
        p["tags"] = tagsets.get(p["name"], [])
        p["fotos"] = cuentas.get(p["name"], 0)
    return render_template_string(
        shell("known", KNOWN_TEMPLATE, head_extra=REFRESH_IMPORT),
        people=people, notfaces=count_not_faces(),
        known_dir=get_library(), all_tags=tags_mod.all_tags(),
        imp=IMPORT_STATE,
        message=request.args.get("msg", ""))


@app.route("/known/import_folder", methods=["POST"])
def known_import_folder():
    """Importa una carpeta como una persona: copia TODAS sus fotos a
    <Biblioteca>/<nombre>/ sin analizar caras. Todo lo de adentro es esa persona."""
    folder = request.form.get("folder", "").strip()
    name = request.form.get("name", "").strip()
    library = get_library()
    if not library:
        return redirect(url_for("known_page",
                                msg="Primero configurá la Biblioteca en Ajustes."))
    try:
        r = import_folder_as_person(library, folder, name)
    except ValueError as e:
        return redirect(url_for("known_page", msg=str(e)))
    except Exception as e:
        return redirect(url_for("known_page", msg="No se pudo importar: " + str(e)))
    quien = name or Path(folder).name
    if r["copiadas"]:
        return redirect(url_for("known_page",
            msg=f"Importadas {r['copiadas']} foto(s) como «{quien}» (sin analizar caras)."))
    # 0 copiadas: explico POR QUÉ, con lo que realmente vi en la carpeta
    if r["dup"]:
        return redirect(url_for("known_page",
            msg=f"«{quien}» ya tenía esas {r['dup']} foto(s): no había nada nuevo."))
    if r["vistos"] == 0:
        return redirect(url_for("known_page",
            msg="Esa carpeta está vacía (o no tengo permiso para leerla)."))
    detalle = ", ".join(f"{ext} ({n})" for ext, n in r["otras"][:8])
    return redirect(url_for("known_page",
        msg=(f"Vi {r['vistos']} archivo(s) pero ninguno es imagen que pueda usar. "
             f"Encontré: {detalle}. Soporto: jpg, jpeg, png, webp, gif, bmp, jfif, avif. "
             "(Ojo: Windows oculta las extensiones, un archivo puede no ser el formato que parece.)")))


SETTINGS_TEMPLATE = """
<h1>Ajustes <span class="hint-ico" title="La Biblioteca: la carpeta fija donde viven tus personas ordenadas (Biblioteca/&lt;persona&gt;/). El Origen del análisis se elige en cada corrida desde &quot;Ordenar por rostro&quot;, porque cambia seguido.">ⓘ</span></h1>

{% if message %}<div class="msg">{{ message }}</div>{% endif %}

<div class="panel">
  <form method="post" action="{{ url_for('settings_save') }}">
    <div class="row">
      <label>📚 Biblioteca: <span class="hint-ico" title="Es una sola y la usan todos: el organizador y el buscador mueven acá; la galería, el import, el corrector y el separador leen de acá.">ⓘ</span></label>
      <input class="path" type="text" name="library" value="{{ library }}"
             placeholder="Carpeta con una subcarpeta por persona">
      <button type="button" onclick="browse('library')">📂 Elegir carpeta…</button>
    </div>

    <button class="primary">💾 Guardar</button>
  </form>
</div>

<div class="panel">
  <div class="counter">💾 Backup <span class="hint-ico" title="Baja un ZIP con todo tu trabajo: etiquetas, pies y comentarios, conversiones, la base de caras y los ajustes. NO incluye las fotos (esas ya las tenés). Guardalo cada tanto por las dudas: si algo se corrompe, restaurás pegando los .json de vuelta en data/.">ⓘ</span></div>
  <a class="primary" style="display:inline-block; text-decoration:none"
     href="{{ url_for('backup') }}">💾 Descargar backup (ZIP)</a>
</div>

<div class="panel">
  <div class="counter">🎬 Duplicados — videos <span class="hint-ico" title="El buscador de duplicados también revisa videos, pero solo como duplicado exacto (el mismo archivo, ej: un clip de WhatsApp re-descargado) y hasta cierto tamaño. Uno gigante tarda una eternidad leyéndolo y casi nunca está repetido, así que se saltea.">ⓘ</span></div>
  <form method="post" action="{{ url_for('settings_save') }}">
    <input type="hidden" name="library" value="{{ library }}">
    <div class="row">
      <label>Revisar videos de hasta:</label>
      <input type="number" name="dup_video_max_mb" min="1" max="100000"
             value="{{ dup_video_max_mb }}" style="width:6rem">
      <span class="meta">MB — los más grandes se saltean</span>
    </div>
    <button class="primary">💾 Guardar</button>
  </form>
</div>

<div class="panel">
  <div class="counter">💬 Comentarios y pie de foto <span class="hint-ico" title="El programa trae 62 pies de fábrica que quedan mezclados con los tuyos y hacen que varíe mucho antes de caer en uno propio. Con esto solo se usan los que escribiste vos, y si no hay ninguno para esa foto te avisa en vez de inventar uno estándar.">ⓘ</span> <span class="hint-ico" title="Cuando abrís una foto de la galería, la IA le escribe un pie y unos comentarios como si fueran de seguidores. Acá elegís con qué onda.">ⓘ</span></div>
  <form method="post" action="{{ url_for('settings_save') }}">
    <input type="hidden" name="library" value="{{ library }}">

    <div class="row">
      <label><input type="checkbox" name="solo_moldes_propios" value="1"
             {% if solo_moldes_propios %}checked{% endif %}> Solo mis pies</label>
    </div>

    <div class="row">
      <label>Vos sos: <span class="hint-ico" title="Para que los vínculos salgan bien en los pies: con {relacion}, si en el perfil dice prima te va a decir «primo» o «prima» según esto. Y da vuelta los que cambian: tu mamá te dice «hijo»/«hija», tu sobrina te dice «tío»/«tía». Sin definir, usa el vínculo tal cual.">ⓘ</span></label>
      <select name="genero_duenio">
        <option value="" {% if not genero_duenio %}selected{% endif %}>(sin definir)</option>
        <option value="m" {% if genero_duenio == 'm' %}selected{% endif %}>hombre</option>
        <option value="f" {% if genero_duenio == 'f' %}selected{% endif %}>mujer</option>
      </select>
    </div>

    {# el pie y los comentarios corrían clavados al 4B: la config ya se leía
       (post_model) pero no había con qué elegirla. Acá está el enchufe. #}
    <div class="row">
      <label>Modelo del pie y comentarios: <span class="hint-ico" title="Los de 7B escriben bastante mejor que el 4B de fábrica, pero en CPU tardan como el doble por foto. Si el pie te sale sin sentido, probá con Qwen2.5-VL-7B-NSFW-Caption: además de ser más grande, está tuneado justo para esto. Los 🔞 son sin censura (no te lavan las frases).">ⓘ</span></label>
      <select name="post_model">
        <option value="">(el de fábrica: {{ default_model }})</option>
        {% for m in vlm_models %}
        <option value="{{ m }}" {% if m == post_model %}selected{% endif %}>{% if m in vlm_crudos %}🔞 {% else %}👁 {% endif %}{{ m }}</option>
        {% endfor %}
      </select>
    </div>

    <div style="margin:.6rem 0">
      <label>Qué tan subido: <b id="crud-lbl">{{ crudeness_labels[crudeness] }}</b> <span class="hint-ico" title="Vale para el pie y para los comentarios. De explícito para arriba, un modelo con filtros te lava las frases: elegí el abliterated en el desplegable del visor.">ⓘ</span></label>
      <input type="range" name="crudeness" min="0" max="{{ crudeness_labels|length - 1 }}"
             step="1" value="{{ crudeness }}" style="width:100%;margin:.3rem 0"
             oninput="document.getElementById('crud-lbl').textContent = {{ crudeness_labels|tojson }}[this.value]">
      <div class="row" style="justify-content:space-between;margin:0">
        {% for l in crudeness_labels %}<span class="meta">{{ l }}</span>{% endfor %}
      </div>
    </div>

    <div class="row">
      <label>Tono:</label>
      <select name="comment_tone">
        {% for k, v in tones.items() %}
        <option value="{{ k }}" {% if k == comment_tone %}selected{% endif %}>{{ k }} — {{ v }}</option>
        {% endfor %}
      </select>
    </div>
    <input type="hidden" name="guardar_post" value="1">
    <div class="row">
      <label><input type="checkbox" name="diminutivos" value="1"
             {% if diminutivos %}checked{% endif %}> Hablar en diminutivo</label>
      <span class="meta">"acá con friito", "comiendo una sopita", "casi desnudita".
        En el pie y en los comentarios, con las palabras de <code>DIMINUTIVOS</code>
        en <code>app/posts.py</code>: agregá las que uses. Los piropos
        (<code>PIROPOS</code>: mamá→mamita) van solo en los comentarios y solo a
        adultos.</span>
    </div>
    <div class="row">
      <label>@usuarios: <span class="hint-ico" title="Los nombres los pongo yo de esta lista, no los inventa la IA: un modelo chico devuelve &quot;@usuario1&quot;, &quot;@usuario2&quot; y queda feo. Vaciá el campo para volver a los de fábrica.">ⓘ</span></label>
      <input class="path" type="text" name="comment_users" value="{{ comment_users }}"
             placeholder="separados por coma">
    </div>
    <button class="primary">💾 Guardar</button>
  </form>
</div>

<div class="panel">
  <div class="counter">📝 Textos y filtros de los posts <span class="hint-ico" title="Para afinar el pie y los comentarios sin tocar código. Todo lo de acá se suma a lo de fábrica; vacío = queda como está.">ⓘ</span></div>
  <form method="post" action="{{ url_for('settings_save') }}">
    <input type="hidden" name="guardar_textos" value="1">

    <div style="margin:.6rem 0">
      <label>➕ Instrucción extra para el PIE de foto</label>
      <textarea class="path" name="extra_pie" rows="2" style="width:100%"
        placeholder="ej: que sea más corto y vulgar. que no hable del fondo.">{{ extra_pie }}</textarea>
      <p class="meta">Se le agrega al final del prompt del pie. No toca lo de fábrica: lo complementa.</p>
    </div>

    <div style="margin:.6rem 0">
      <label>➕ Instrucción extra para los COMENTARIOS</label>
      <textarea class="path" name="extra_com" rows="2" style="width:100%"
        placeholder="ej: que alguno pregunte por la ropa. que sean más largos.">{{ extra_com }}</textarea>
    </div>

    <div style="margin:.6rem 0; border-top:1px solid var(--line); padding-top:.6rem">
      <label>🧩 Prompt para copiar desde una foto <span class="hint-ico" title="Desde el ✍ de una foto, el botón &quot;📋 Copiar prompt&quot; copia este texto reemplazando las variables. Para pegarlo en la IA que quieras y traerte los pies. · $etiquetas → las de esa foto: &quot;playa&quot;, &quot;sol&quot;, &quot;bikini&quot; · $perfil → las del perfil: &quot;mujer&quot;, &quot;adulto&quot;, &quot;hermana&quot; · $vinculo → lo que es ella para vos: prima · $relacion → cómo te llama a vos: primo (necesita tu género)">ⓘ</span></label>
      <textarea class="path" name="prompt_externo" rows="4" style="width:100%"
        placeholder='ej: Escribí 3 pies de foto en primera persona, español rioplatense, usando SOLO estas palabras: $etiquetas'>{{ prompt_externo }}</textarea>
    </div>

    <details class="lote"><summary>Filtros por palabras (avanzado)</summary>
      <div style="margin:.5rem 0">
        <label>🔞 Palabras que marcan MENOR (además de las de fábrica) <span class="hint-ico" title="Suma a: {{ def_menor }}. Con cualquiera, la crudeza se fuerza a sano y los comentadores que desean quedan afuera. Estas no se pueden quitar, solo agregar.">ⓘ</span></label>
        <textarea class="path" name="f_menor" rows="2" style="width:100%"
          placeholder="una por línea… ej: sobrino">{{ f_menor }}</textarea>
      </div>
      <div style="margin:.5rem 0">
        <label>🗣 Correcciones tú→vos (una por línea, formato <code>tienes=tenés</code>)</label>
        <textarea class="path" name="f_tuteo" rows="2" style="width:100%"
          placeholder="querés=queres">{{ f_tuteo }}</textarea>
      </div>
      <div style="margin:.5rem 0">
        <label>🔁 Muletillas que se ignoran al comparar comentarios repetidos</label>
        <textarea class="path" name="f_vacias" rows="2" style="width:100%"
          placeholder="una por línea">{{ f_vacias }}</textarea>
        <p class="meta">Palabras tan comunes que no cuentan para decidir si dos comentarios se parecen.</p>
      </div>
      <div style="margin:.5rem 0">
        <label>🔤 Diminutivos y piropos (formato <code>palabra=forma1, forma2</code>)</label>
        <textarea class="path" name="f_diminutivos" rows="2" style="width:100%"
          placeholder="sopa=sopita">{{ f_diminutivos }}</textarea>
        <textarea class="path" name="f_piropos" rows="2" style="width:100%;margin-top:.3rem"
          placeholder="piropos (solo a adultos): reina=reinita, reinaza">{{ f_piropos }}</textarea>
        <p class="meta">Solo actúan con el tilde "Hablar en diminutivo" de arriba. Los piropos, solo a adultos.</p>
      </div>
    </details>

    <details class="lote"><summary>Reescribir los 5 niveles de crudeza</summary>
      <p class="meta">Vacío = el texto de fábrica de ese nivel.</p>
      {% for lbl, val, ph in crudeness_rows %}
      <div class="row" style="margin:.25rem 0">
        <label style="min-width:5rem">{{ lbl }}:</label>
        <input class="path" name="f_crud_{{ loop.index0 }}" value="{{ val }}" placeholder="{{ ph }}">
      </div>
      {% endfor %}
    </details>

    <button class="primary">💾 Guardar textos y filtros</button>
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
    # los diccionarios de override (palabra: [formas]) se muestran como texto
    def _mapa_a_texto(d):
        out = []
        for k, v in (d or {}).items():
            formas = v if isinstance(v, list) else [v]
            out.append(f"{k}={', '.join(formas)}")
        return "\n".join(out)
    ov_crud = cfg.get("f_crudeness") or []
    crudeness_rows = [
        (clave, ov_crud[i] if i < len(ov_crud) else "", txt)
        for i, (clave, txt) in enumerate(posts_mod.CRUDENESS)]
    return render_template_string(
        shell("settings", SETTINGS_TEMPLATE),
        cfg=cfg, library=library, n_people=n_people,
        tones=posts_mod.TONES,
        comment_tone=cfg.get("comment_tone", posts_mod.DEFAULT_TONE),
        comment_users=", ".join(cfg.get("comment_users") or posts_mod.DEFAULT_USERS),
        crudeness_labels=[k for k, _ in posts_mod.CRUDENESS],
        crudeness=cfg.get("crudeness", posts_mod.CRUDENESS_DEFAULT),
        diminutivos=bool(cfg.get("diminutivos")),
        # con qué modelo se escriben el pie y los comentarios (vacío = el de fábrica)
        post_model=cfg.get("post_model", ""),
        genero_duenio=cfg.get("genero_duenio", ""),
        solo_moldes_propios=bool(cfg.get("solo_moldes_propios")),
        default_model=vlm.DEFAULT_MODEL,
        vlm_models=vlm.list_models(),
        vlm_crudos=vlm.modelos_descensurados(),
        dup_video_max_mb=cfg.get("dup_video_max_mb", DEFAULT_VIDEO_MAX_MB),
        # textos y filtros
        extra_pie=cfg.get("post_extra_pie", ""),
        prompt_externo=cfg.get("prompt_externo", ""),
        extra_com=cfg.get("post_extra_com", ""),
        f_menor="\n".join(cfg.get("f_menor_extra", [])),
        f_tuteo="\n".join(cfg.get("f_tuteo_extra", [])),
        f_vacias="\n".join(cfg.get("f_vacias_extra", [])),
        f_diminutivos=_mapa_a_texto(cfg.get("f_diminutivos_extra")),
        f_piropos=_mapa_a_texto(cfg.get("f_piropos_extra")),
        def_menor=", ".join(sorted(posts_mod._MENOR)),
        crudeness_rows=crudeness_rows,
        message=request.args.get("msg", ""))


@app.route("/backup")
def backup():
    """Respalda TODO tu trabajo (los .json de data/: etiquetas, pies, conversiones,
    base de caras, etc.) en un ZIP descargable. No incluye las fotos (esas son
    tuyas y viven en la Biblioteca); sí lo irremplazable que genera el programa."""
    import datetime
    import io
    import zipfile
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(DATA_DIR.glob("*.json")):
            z.write(p, p.name)
    buf.seek(0)
    fn = "photo-sorter-backup-" + datetime.datetime.now().strftime("%Y%m%d-%H%M") + ".zip"
    return send_file(buf, mimetype="application/zip", as_attachment=True, download_name=fn)


@app.route("/settings/save", methods=["POST"])
def settings_save():
    """Guarda la Biblioteca (una sola para todo). El Origen se elige por
    análisis en la página principal."""
    cfg = load_config()
    # La biblioteca se toca SOLO si el form la mandó con una ruta. Antes se
    # pisaba SIEMPRE (set_library con "" si no venía), así que guardar el
    # formulario de textos —que no manda `library`— borraba la biblioteca
    # entera. Ahora cada campo se toca solo si vino, la biblioteca incluida.
    library = request.form.get("library", "").strip()
    if library:
        if not Path(library).is_dir():
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
    if "guardar_post" in request.form:      # el form de los comentarios
        cfg["diminutivos"] = request.form.get("diminutivos") == "1"
        cfg["solo_moldes_propios"] = request.form.get("solo_moldes_propios") == "1"
    if "genero_duenio" in request.form:     # tu género: da vuelta los vínculos
        g = request.form.get("genero_duenio", "").strip().lower()
        cfg["genero_duenio"] = g if g in ("m", "f") else ""
    if "post_model" in request.form:        # con qué modelo se escribe el pie
        # vacío = el de fábrica. Solo se acepta un modelo que EXISTA: si no, un
        # nombre mal tipeado deja el pie roto y el error recién aparece al generar.
        elegido = request.form["post_model"].strip()
        cfg["post_model"] = elegido if elegido in set(vlm.list_models()) else ""
    if "crudeness" in request.form:
        try:
            cfg["crudeness"] = max(0, min(int(request.form["crudeness"]),
                                          len(posts_mod.CRUDENESS) - 1))
        except ValueError:
            pass
    if "dup_video_max_mb" in request.form:      # límite de tamaño de video del duplicador
        try:
            cfg["dup_video_max_mb"] = max(1, min(int(request.form["dup_video_max_mb"]), 100000))
        except ValueError:
            pass

    if "guardar_textos" in request.form:      # el form de textos y filtros
        f = request.form
        cfg["post_extra_pie"] = f.get("extra_pie", "").strip()
        cfg["post_extra_com"] = f.get("extra_com", "").strip()
        cfg["prompt_externo"] = f.get("prompt_externo", "").strip()
        # listas: una palabra/línea, sin vacías ni duplicados
        def _lineas(campo):
            vistas, out = set(), []
            for x in f.get(campo, "").splitlines():
                x = x.strip()
                if x and x.lower() not in vistas:
                    vistas.add(x.lower())
                    out.append(x)
            return out
        cfg["f_menor_extra"] = _lineas("f_menor")
        cfg["f_tuteo_extra"] = [x for x in _lineas("f_tuteo") if "=" in x]
        cfg["f_vacias_extra"] = _lineas("f_vacias")
        # diccionarios: "palabra=forma1, forma2" por línea
        def _mapa(campo):
            d = {}
            for linea in f.get(campo, "").splitlines():
                if "=" not in linea:
                    continue
                k, v = linea.split("=", 1)
                k = k.strip()
                formas = [w.strip() for w in v.split(",") if w.strip()]
                if k and formas:
                    d[k] = formas
            return d
        cfg["f_diminutivos_extra"] = _mapa("f_diminutivos")
        cfg["f_piropos_extra"] = _mapa("f_piropos")
        # los 5 niveles de crudeza: "" en un nivel = el de fábrica
        crud = [f.get(f"f_crud_{i}", "").strip()
                for i in range(len(posts_mod.CRUDENESS))]
        cfg["f_crudeness"] = crud if any(crud) else []

    save_config(cfg)
    return redirect(url_for("settings_page", msg="Ajustes guardados."))


@app.route("/notfaces/clear", methods=["POST"])
def notfaces_clear():
    clear_not_faces()
    return redirect(url_for("known_page", msg="Lista negra vaciada."))



FINDER_TEMPLATE = """
<h1>Buscar persona <span class="hint-ico" title="Elegí una persona conocida y una carpeta: se revisan todas sus fotos y videos buscando esa cara. Después podés mover los encontrados a la carpeta de la persona (la de las galerías).">ⓘ</span></h1>

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
        data-confirm="Se moverán los tildados a la carpeta de {{ state.name }}. Los que destildaste se descartan de esta lista. ¿Continuar?">
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
  <div class="counter">{{ state.by_person|length }} conocido(s) encontrados en {{ state.scanned }} archivo(s) revisados <span class="hint-ico" title="Un archivo donde aparecen dos conocidos figura en los dos grupos: al moverlo se copia a la carpeta de cada uno y el original desaparece recién con el último.">ⓘ</span></div>
  {% if not state.by_person %}
  <p class="meta">Ningún conocido apareció en esa carpeta con el rigor actual. Probá subir el deslizador hacia &quot;permisivo&quot; y buscar de nuevo.</p>
  {% else %}
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
        procesadas.olvidar(name)          # el registro de "ya pasó por la IA"
    return redirect(url_for("known_page", msg=f"Se olvidó a {name}."))


@app.route("/person/<name>/vinculo", methods=["POST"])
def person_vinculo(name):
    """Fija el vínculo de la persona (hermana, prima…) como etiqueta de perfil.

    Reemplaza el vínculo anterior si había, y NO toca el resto de sus etiquetas:
    el vínculo es un campo, no una etiqueta más de la lista."""
    nuevo = (request.form.get("vinculo") or "").strip()
    actuales = tags_mod.get_person(name)
    # sacar el vínculo viejo (si tenía) y poner el nuevo
    viejo = moldes_mod.relacion_de(actuales)
    quedan = [t for t in actuales if t != viejo] if viejo else list(actuales)
    if nuevo:
        quedan.append(nuevo)
    tags_mod.set_person(name, quedan)
    return redirect(url_for("person_page", name=name))


@app.route("/person/<name>/renombrar", methods=["POST"])
def person_renombrar(name):
    """Renombra a la persona: mueve su carpeta y migra TODO lo asociado.

    Sin esto, equivocarse al importar obligaba a borrar y rehacer, perdiendo las
    etiquetas. Se valida que el destino no exista para no fusionar dos personas
    sin querer."""
    nuevo = (request.form.get("nuevo") or "").strip()
    if not nuevo or nuevo == name:
        return redirect(url_for("person_page", name=name))
    if any(c in nuevo for c in '\\/:*?"<>|'):
        return redirect(url_for("person_page", name=name,
                                msg="Ese nombre tiene caracteres que no valen para una carpeta."))
    library = get_library()
    origen, destino = Path(library) / name, Path(library) / nuevo
    if destino.exists():
        return redirect(url_for("person_page", name=name,
                                msg=f"Ya existe una carpeta «{nuevo}»: elegí otro nombre."))
    try:
        if origen.is_dir():
            origen.rename(destino)
    except OSError as e:
        return redirect(url_for("person_page", name=name, msg=f"No se pudo mover la carpeta: {e}"))
    # la carpeta ya se movió: migrar todo lo que cuelga del nombre
    db = load_known()
    if name in db:
        db[nuevo] = db.pop(name)
        save_known(db)
        viejo_thumb = KNOWN_THUMBS / f"{safe_name(name)}.jpg"
        if viejo_thumb.is_file():
            viejo_thumb.rename(KNOWN_THUMBS / f"{safe_name(nuevo)}.jpg")
    tags_mod.rename_person(name, nuevo)
    posts_mod.rename_person(name, nuevo)
    person_meta.rename(name, nuevo)
    for rel, info in procesadas.de_persona(name).items():
        procesadas.marcar(nuevo, rel, info.get("n", 0))
    procesadas.olvidar(name)
    return redirect(url_for("person_page", name=nuevo, msg=f"Ahora se llama {nuevo}."))


@app.route("/person/<name>/perfil", methods=["POST"])
def person_perfil(name):
    """Configuración del perfil (modal ⚙): bio + voz, en una sola escritura."""
    try:
        bio, voice = person_meta.set_profile(
            name,
            request.form.get("bio", ""),
            request.form.get("voice", ""))
    except Exception as e:
        return {"ok": False, "error": str(e)}
    return {"ok": True, "bio": bio, "voice": voice}


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

{# el lote corriendo: se muestra en la página de la persona que lo lanzó #}
<div class="panel" id="lotebox" style="display:none">
  <div class="row" style="justify-content:space-between">
    <b>✨ Escribiendo posts de <span id="lote-name"></span></b>
    <button class="secondary" onclick="loteStop()" id="lote-stop">■ Parar</button>
  </div>
  <progress id="lote-bar" value="0" max="1" style="width:100%"></progress>
  <p class="meta">
    <b><span id="lote-cur">0</span>/<span id="lote-tot">0</span></b> ·
    falta <b id="lote-eta">calculando…</b> · lleva <span id="lote-lleva">0 s</span><br>
    <span id="lote-foto" class="fn"></span><br>
    ✓ <span id="lote-hechas">0</span> hechas ·
    ⏭ <span id="lote-salt">0</span> ya tenían ·
    ✕ <span id="lote-fall">0</span> fallaron
  </p>
  <p class="meta" id="lote-err" style="color:var(--red)"></p>
  <div style="margin-top:.5rem">
    <span class="meta">Últimos, para que veas cómo van saliendo (si vienen mal, pará):</span>
    <ul id="lote-ultimos" class="meta" style="margin:.2rem 0 0;padding-left:1.1rem"></ul>
    <span class="hint-ico" title="Podés cerrar esta página o apagar la pantalla: sigue igual. Lo hecho queda guardado foto por foto, así que cortarlo no pierde nada.">ⓘ</span>
  </div>
</div>

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

<div class="pcard">
  <!-- Columna izquierda: portada + fechas -->
  <div class="pcard-cover">
    <div class="cover-box" title="Portada — tocá ✏️ para elegir y recortar">
      {% if has_thumb %}
        <img src="{{ url_for('known_thumb', name=name) }}?v={{ cache_bust }}" alt="portada de {{ name }}">
      {% else %}
        <div class="noimg">👤</div>
      {% endif %}
      {% if display %}
      <button class="secondary iconbtn cover-edit" onclick="coverAbrir()"
              title="Editar portada: elegí una foto y recortala">✏️</button>
      {% endif %}
    </div>

    <!-- Editor de portada: elegir foto + recortar -->
    <style>
      .ce-overlay { position:fixed; inset:0; z-index:800; background:rgba(0,0,0,.85);
        display:flex; align-items:center; justify-content:center; }
      .ce-box { background:var(--pan); border:1px solid var(--bd); border-radius:12px;
        padding:1rem; max-width:min(680px,92vw); max-height:90vh; overflow:auto; }
      .ce-top { display:flex; justify-content:space-between; align-items:center; margin-bottom:.6rem; }
      .ce-close { background:none; border:0; color:var(--mut); font-size:1.3rem; cursor:pointer; }
      .ce-grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(92px,1fr)); gap:.4rem; }
      .ce-grid img { width:100%; aspect-ratio:1; object-fit:cover; border-radius:8px; cursor:pointer;
        border:2px solid transparent; }
      .ce-grid img:hover { border-color:var(--acc); }
      .ce-stage { position:relative; display:inline-block; max-width:100%; user-select:none; line-height:0; }
      .ce-stage img { display:block; max-width:100%; max-height:60vh; }
      .ce-crop { position:absolute; box-sizing:border-box; border:2px solid #fff;
        box-shadow:0 0 0 9999px rgba(0,0,0,.55); cursor:move; }
      .ce-handle { position:absolute; right:-8px; bottom:-8px; width:16px; height:16px;
        background:#fff; border-radius:50%; cursor:nwse-resize; }
      .ce-actions { display:flex; gap:.5rem; align-items:center; margin-top:.7rem; }
    </style>
    <div id="cover-modal" class="ce-overlay" style="display:none"
         onclick="if(event.target===this)coverCerrar()">
      <div class="ce-box">
        <div class="ce-top">
          <b id="ce-title">Elegí una foto para la portada</b>
          <button class="ce-close" onclick="coverCerrar()">✕</button>
        </div>
        <div id="ce-grid" class="ce-grid"></div>
        <div id="ce-crop-wrap" style="display:none">
          <div class="ce-stage" id="ce-stage">
            <img id="ce-img" alt="">
            <div class="ce-crop" id="ce-crop"><div class="ce-handle" id="ce-handle"></div></div>
          </div>
          <div class="ce-actions">
            <button class="secondary" onclick="coverVolver()">‹ Elegir otra</button>
            <button class="primary" onclick="coverGuardar()">💾 Guardar portada</button>
            <span class="meta" id="ce-st"></span>
          </div>
          <p class="meta" style="margin-top:.4rem">Arrastrá el cuadro para mover · tirá de la
            esquina (○) para agrandar o achicar.</p>
        </div>
      </div>
    </div>
    <script>
    (function() {
      const PERSONA = {{ name|tojson }};
      const ceg = id => document.getElementById(id);
      let ceRel = null, ceCrop = {x:0, y:0, s:0}, ceDrag = null;

      const COVER_SRC = {{ cover_src|tojson }};   // {rel,left,top,side} o null
      function urlFoto(rel) {
        return '/person_photo/' + encodeURIComponent(PERSONA) + '/' +
               String(rel).split('/').map(encodeURIComponent).join('/');
      }
      // Abrir el editor: si ya sabemos de qué foto salió la portada, vamos DIRECTO
      // a recortar esa (con su recorte anterior). Si no, mostramos el elegidor.
      window.coverAbrir = function() {
        if (COVER_SRC && COVER_SRC.rel) {
          coverEditar(COVER_SRC.rel, urlFoto(COVER_SRC.rel), COVER_SRC);
          return;
        }
        coverPicker();
      };
      window.coverPicker = function() {
        ceg('cover-modal').style.display = 'flex';
        ceg('ce-crop-wrap').style.display = 'none';
        ceg('ce-grid').style.display = '';
        ceg('ce-title').textContent = 'Elegí una foto para la portada';
        const grid = ceg('ce-grid'); grid.innerHTML = '<span class="meta">cargando…</span>';
        fetch('/probador/fotos/' + encodeURIComponent(PERSONA)).then(r => r.json()).then(d => {
          grid.innerHTML = '';
          (d.fotos || []).forEach(f => {
            const im = document.createElement('img'); im.src = f.thumb; im.title = f.rel;
            im.onclick = () => coverEditar(f.rel, f.url);
            grid.appendChild(im);
          });
          if (!(d.fotos || []).length) grid.innerHTML = '<span class="meta">esta persona no tiene fotos.</span>';
        }).catch(e => grid.innerHTML = '<span class="meta">' + e + '</span>');
      };
      window.coverCerrar = function() { ceg('cover-modal').style.display = 'none'; };
      window.coverVolver = function() { coverPicker(); };

      // Abre el recortador en UNA foto concreta. `box` (opcional) es el recorte
      // guardado, en píxeles reales: se reescala a lo que se muestra en pantalla.
      window.coverEditar = function(rel, url, box) {
        ceRel = rel;
        ceg('cover-modal').style.display = 'flex';
        ceg('ce-grid').style.display = 'none';
        ceg('ce-crop-wrap').style.display = '';
        ceg('ce-title').textContent = 'Recortá la portada';
        const img = ceg('ce-img');
        img.onload = () => {
          const w = img.clientWidth, h = img.clientHeight;
          if (box && box.side) {                      // restaurar el recorte previo
            const esc = w / (img.naturalWidth || w);
            ceCrop = {x: Math.round(box.left * esc), y: Math.round(box.top * esc),
                      s: Math.round(box.side * esc)};
          } else {                                     // por defecto: cuadrado centrado
            const s = Math.round(Math.min(w, h) * 0.85);
            ceCrop = {x: Math.round((w - s) / 2), y: Math.round((h - s) / 2), s: s};
          }
          ceClamp(); cePintar();
        };
        img.src = url;
      };
      function cePintar() {
        const c = ceg('ce-crop');
        c.style.left = ceCrop.x + 'px'; c.style.top = ceCrop.y + 'px';
        c.style.width = ceCrop.s + 'px'; c.style.height = ceCrop.s + 'px';
      }
      function ceClamp() {
        const img = ceg('ce-img'), w = img.clientWidth, h = img.clientHeight;
        ceCrop.s = Math.max(32, Math.min(ceCrop.s, Math.min(w, h)));
        ceCrop.x = Math.max(0, Math.min(ceCrop.x, w - ceCrop.s));
        ceCrop.y = Math.max(0, Math.min(ceCrop.y, h - ceCrop.s));
      }
      ceg('ce-crop').addEventListener('mousedown', e => {
        if (e.target === ceg('ce-handle')) ceDrag = {modo:'resize', sx:e.clientX, sy:e.clientY, s0:ceCrop.s};
        else ceDrag = {modo:'move', sx:e.clientX, sy:e.clientY, x0:ceCrop.x, y0:ceCrop.y};
        e.preventDefault();
      });
      window.addEventListener('mousemove', e => {
        if (!ceDrag) return;
        if (ceDrag.modo === 'move') { ceCrop.x = ceDrag.x0 + (e.clientX - ceDrag.sx); ceCrop.y = ceDrag.y0 + (e.clientY - ceDrag.sy); }
        else ceCrop.s = ceDrag.s0 + Math.max(e.clientX - ceDrag.sx, e.clientY - ceDrag.sy);
        ceClamp(); cePintar();
      });
      window.addEventListener('mouseup', () => { ceDrag = null; });

      window.coverGuardar = function() {
        if (!ceRel) return;
        const img = ceg('ce-img');
        const scale = img.naturalWidth / img.clientWidth;    // pantalla -> pixeles reales
        const body = new URLSearchParams({rel: ceRel,
          left: Math.round(ceCrop.x * scale), top: Math.round(ceCrop.y * scale),
          side: Math.round(ceCrop.s * scale)});
        ceg('ce-st').textContent = ' guardando…';
        fetch('/person/' + encodeURIComponent(PERSONA) + '/cover_crop', {method:'POST',
          headers:{'Content-Type':'application/x-www-form-urlencoded'}, body: body})
          .then(r => r.json()).then(d => {
            if (!d.ok) { ceg('ce-st').textContent = ' ' + (d.error || 'error'); return; }
            const box = document.querySelector('.cover-box');
            let cov = box.querySelector('img');
            if (!cov) { const ni = box.querySelector('.noimg'); if (ni) ni.remove();
                        cov = document.createElement('img'); box.insertBefore(cov, box.firstChild); }
            cov.src = d.url + '?v=' + Date.now();
            coverCerrar();
          }).catch(e => ceg('ce-st').textContent = ' ' + e);
      };
    })();
    </script>

    <!-- Gestor de etiquetas: las del perfil + todas las de las fotos con recuento -->
    <style>
      .tm-overlay { position:fixed; inset:0; z-index:820; background:rgba(0,0,0,.85);
        display:flex; align-items:center; justify-content:center; }
      .tm-box { background:var(--pan); border:1px solid var(--bd); border-radius:12px;
        padding:1rem; width:min(620px,94vw); max-height:88vh; overflow:auto; }
      .tm-top { display:flex; justify-content:space-between; align-items:center; }
      .tm-sec { margin-top:1rem; }
      .tm-chips { display:flex; flex-wrap:wrap; gap:.35rem; align-items:center; margin-top:.45rem; }
      .tm-row { display:flex; justify-content:space-between; align-items:center; gap:.5rem;
        padding:.32rem .15rem; border-bottom:1px solid var(--line); }
      .tm-n { color:var(--mut); font-size:.78rem; }
      .tm-del { background:none; border:0; color:var(--mut); cursor:pointer; font-size:1rem; }
      .tm-del:hover { color:var(--red); }
      .tm-fav { background:none; border:0; cursor:pointer; font-size:.95rem; padding:0 .1rem;
        line-height:1; }
      .tm-fav:hover { transform:scale(1.2); }
      .tm-in { background:var(--bg); border:1px solid var(--bd); border-radius:6px;
        color:var(--tx); font-size:.85rem; padding:.28rem .55rem; }
    </style>
    <div id="tagmgr" class="tm-overlay" style="display:none"
         onclick="if(event.target===this)cerrarTagMgr()">
      <div class="tm-box">
        <div class="tm-top">
          <b>🗂 Etiquetas de {{ name }}</b>
          <button class="secondary sm" onclick="cerrarTagMgr()">✕</button>
        </div>

        <div class="tm-sec">
          <b style="font-size:.92rem">👤 Del perfil — valen para TODAS sus fotos</b>
          <p class="meta" style="margin:.2rem 0 0">Cada foto las hereda. Si la sacás de acá,
            se va de todas.</p>
          <div class="tm-chips" id="tm-persona"></div>
        </div>

        <div class="tm-sec">
          <b style="font-size:.92rem">🖼 De las fotos — y en cuántas está cada una <span class="hint-ico" title="La ✕ la borra de todas las fotos que la tengan. Sirve para limpiar rápido después de etiquetar en lote con IA.">ⓘ</span></b>
          <input class="tm-in" id="tm-filtro" placeholder="🔍 filtrar…"
                 oninput="tmPintarFotos()" style="width:100%">
          <div id="tm-fotos" style="margin-top:.45rem"></div>
        </div>
        <span class="meta" id="tm-st"></span>
      </div>
    </div>
    <script>
    (function() {
      const tmg = id => document.getElementById(id);
      const TM_DATA = "{{ url_for('person_tags_data', name=name) }}";
      const TM_DEL  = "{{ url_for('person_tags_remove_all', name=name) }}";
      const TM_REN  = "{{ url_for('person_tags_rename_all', name=name) }}";
      const TM_SET  = "{{ url_for('person_tags_person_set', name=name) }}";
      const TM_FAV  = "{{ url_for('person_tags_fav', name=name) }}";
      const TM_GEN  = "{{ url_for('person_tags_promote', name=name) }}";
      let tmPersona = [], tmFotos = [], tmFavPers = [], tmFavGlob = [];

      window.abrirTagMgr = function() {
        tmg('tagmgr').style.display = 'flex';
        tmg('tm-st').textContent = '';
        tmg('tm-persona').innerHTML = '<span class="meta">cargando…</span>';
        tmg('tm-fotos').innerHTML = '';
        fetch(TM_DATA).then(r => r.json()).then(d => {
          tmPersona = d.persona || []; tmFotos = d.fotos || [];
          tmFavPers = d.fav_persona || []; tmFavGlob = d.fav_global || [];
          tmPintarPersona(); tmPintarFotos();
        }).catch(e => { tmg('tm-persona').textContent = '' + e; });
      };
      window.cerrarTagMgr = function() { tmg('tagmgr').style.display = 'none'; };

      function tmGuardarPersona() {
        fetch(TM_SET, {method: 'POST',
          headers: {'Content-Type': 'application/x-www-form-urlencoded'},
          body: new URLSearchParams({tags: JSON.stringify(tmPersona)})})
          .then(r => r.json()).then(d => {
            if (!d.ok) { tmg('tm-st').textContent = ' error'; return; }
            tmPersona = d.tags || [];
            // que el visor abierto lo refleje sin recargar
            try { PERSON_TAGS = tmPersona.slice(); } catch (e) {}
            tmPintarPersona();
            tmg('tm-st').textContent = ' ✓ guardado';
            setTimeout(() => tmg('tm-st').textContent = '', 2000);
          }).catch(e => tmg('tm-st').textContent = ' ' + e);
      }
      function tmPintarPersona() {
        const box = tmg('tm-persona'); box.innerHTML = '';
        if (!tmPersona.length) {
          const s = document.createElement('span'); s.className = 'meta';
          s.textContent = 'ninguna todavía'; box.appendChild(s);
        }
        tmPersona.forEach(t => {
          const c = document.createElement('span'); c.className = 'chip';
          const txt = document.createElement('span'); txt.className = 'chiptxt';
          txt.textContent = t; txt.title = 'clic para editar';
          txt.onclick = () => tmEditarPersona(t, c);
          const x = document.createElement('span'); x.className = 'x'; x.textContent = '×';
          x.title = 'Quitar del perfil (se va de todas sus fotos)';
          x.onclick = () => { tmPersona = tmPersona.filter(v => v !== t); tmGuardarPersona(); };
          c.appendChild(txt); c.appendChild(document.createTextNode(' ')); c.appendChild(x);
          box.appendChild(c);
        });
        const inp = document.createElement('input');
        inp.className = 'tm-in'; inp.placeholder = '+ etiqueta del perfil'; inp.spellcheck = false;
        inp.onkeydown = (e) => {
          if (e.key !== 'Enter') return;
          e.preventDefault();
          const v = inp.value.trim();
          if (v && !tmPersona.includes(v)) { tmPersona.push(v); tmGuardarPersona(); }
          inp.value = '';
        };
        box.appendChild(inp);
      }

      function tmPintarFotos() {
        const box = tmg('tm-fotos'); box.innerHTML = '';
        const q = (tmg('tm-filtro').value || '').trim().toLowerCase();
        const vis = tmFotos.filter(f => !q || f.tag.toLowerCase().includes(q));
        if (!vis.length) {
          box.innerHTML = '<span class="meta">' +
            (tmFotos.length ? 'ninguna coincide con el filtro.' : 'sus fotos no tienen etiquetas propias.') +
            '</span>';
          return;
        }
        vis.forEach(f => {
          const row = document.createElement('div'); row.className = 'tm-row';
          const izq = document.createElement('span');
          izq.textContent = f.tag;
          izq.style.cursor = 'text';
          izq.title = 'clic para renombrarla en TODAS esas fotos';
          izq.onclick = () => tmEditarFoto(f, izq);
          const der = document.createElement('span');
          der.style.display = 'flex'; der.style.alignItems = 'center'; der.style.gap = '.5rem';
          const n = document.createElement('span'); n.className = 'tm-n';
          n.textContent = 'en ' + f.n + ' foto' + (f.n === 1 ? '' : 's');
          // ⭐ favorita de esta galería · 🌐 favorita global · ⬆ hacerla general
          const est = document.createElement('button'); est.className = 'tm-fav';
          const esPers = tmFavPers.includes(f.tag);
          est.textContent = '⭐'; est.style.opacity = esPers ? '1' : '.32';
          est.title = esPers ? 'Favorita de esta galería (clic para sacarla)'
                             : 'Marcar favorita de esta galería';
          est.onclick = () => tmFav(f.tag, 'persona');
          const glo = document.createElement('button'); glo.className = 'tm-fav';
          const esGlo = tmFavGlob.includes(f.tag);
          glo.textContent = '🌐'; glo.style.opacity = esGlo ? '1' : '.32';
          glo.title = esGlo ? 'Favorita global (clic para sacarla)' : 'Marcar favorita global';
          glo.onclick = () => tmFav(f.tag, 'global');
          const gen = document.createElement('button'); gen.className = 'tm-fav';
          gen.textContent = '⬆'; gen.title = 'Hacerla general del perfil (y sacarla de las fotos)';
          gen.onclick = () => tmHacerGeneral(f.tag);
          const x = document.createElement('button'); x.className = 'tm-del'; x.textContent = '✕';
          x.title = 'Borrarla de todas esas fotos';
          x.onclick = () => tmBorrarDeTodas(f.tag, f.n);
          der.appendChild(n); der.appendChild(est); der.appendChild(glo);
          der.appendChild(gen); der.appendChild(x);
          row.appendChild(izq); row.appendChild(der);
          box.appendChild(row);
        });
      }
      function tmBorrarDeTodas(tag, n) {
        // sin confirmación a propósito: es un clic y listo. Si te equivocaste,
        // está el botón flotante "↩ Deshacer" (restaura la etiqueta en esas fotos).
        tmg('tm-st').textContent = ' borrando…';
        fetch(TM_DEL, {method: 'POST',
          headers: {'Content-Type': 'application/x-www-form-urlencoded'},
          body: new URLSearchParams({tag: tag})})
          .then(r => r.json()).then(d => {
            if (!d.ok) { tmg('tm-st').textContent = ' ' + (d.error || 'error'); return; }
            tmFotos = tmFotos.filter(f => f.tag !== tag);
            // sacarla también de la memoria del visor, para no recargar
            try {
              Object.keys(PHOTO_TAGS).forEach(k => {
                PHOTO_TAGS[k] = (PHOTO_TAGS[k] || []).filter(t => t !== tag);
              });
            } catch (e) {}
            tmPintarFotos();
            tmg('tm-st').textContent = ' ✓ quitada de ' + d.quitadas + ' foto(s) · ↩ deshacible';
            if (window.refrescarUndo) window.refrescarUndo();   // que aparezca el botón ya
            setTimeout(() => tmg('tm-st').textContent = '', 3000);
          }).catch(e => tmg('tm-st').textContent = ' ' + e);
      }
      function tmFav(tag, ambito) {   // ⭐ de esta galería o 🌐 global
        fetch(TM_FAV, {method: 'POST',
          headers: {'Content-Type': 'application/x-www-form-urlencoded'},
          body: new URLSearchParams({tag: tag, ambito: ambito})})
          .then(r => r.json()).then(d => {
            if (!d.ok) { tmg('tm-st').textContent = ' error'; return; }
            tmFavPers = d.fav_persona || []; tmFavGlob = d.fav_global || [];
            // que el visor use las favoritas nuevas sin recargar
            try { FAVS = tmFavPers.concat(tmFavGlob.filter(t => !tmFavPers.includes(t))); } catch (e) {}
            tmPintarFotos();
          }).catch(e => tmg('tm-st').textContent = ' ' + e);
      }
      function tmHacerGeneral(tag) {  // ⬆ pasa la etiqueta al perfil y la saca de las fotos
        tmg('tm-st').textContent = ' haciéndola general…';
        fetch(TM_GEN, {method: 'POST',
          headers: {'Content-Type': 'application/x-www-form-urlencoded'},
          body: new URLSearchParams({tag: tag})})
          .then(r => r.json()).then(d => {
            if (!d.ok) { tmg('tm-st').textContent = ' ' + (d.error || 'error'); return; }
            tmPersona = d.persona || tmPersona;
            try {
              PERSON_TAGS = tmPersona.slice();
              Object.keys(PHOTO_TAGS).forEach(k => {
                PHOTO_TAGS[k] = (PHOTO_TAGS[k] || []).filter(t => t !== tag);
              });
            } catch (e) {}
            tmg('tm-st').textContent = ' ✓ general · quitada de ' + d.quitadas + ' foto(s) · ↩ deshacible';
            if (window.refrescarUndo) window.refrescarUndo();
            setTimeout(() => tmg('tm-st').textContent = '', 3500);
            tmRecargar();
          }).catch(e => tmg('tm-st').textContent = ' ' + e);
      }
      function tmRecargar() {         // recuentos frescos (tras renombrar pueden fusionarse)
        return fetch(TM_DATA).then(r => r.json()).then(d => {
          tmPersona = d.persona || []; tmFotos = d.fotos || [];
          tmFavPers = d.fav_persona || []; tmFavGlob = d.fav_global || [];
          tmPintarPersona(); tmPintarFotos();
        });
      }
      // Renombrar una etiqueta de foto en TODAS las fotos que la tengan
      function tmEditarFoto(f, spanEl) {
        const inp = document.createElement('input');
        inp.className = 'tm-in'; inp.value = f.tag; inp.spellcheck = false;
        spanEl.replaceWith(inp); inp.focus(); inp.select();
        let cerrado = false;
        const cancelar = () => { if (!cerrado) { cerrado = true; tmPintarFotos(); } };
        const commit = () => {
          if (cerrado) return;
          const v = inp.value.trim();
          if (!v || v === f.tag) { cancelar(); return; }
          cerrado = true;
          tmg('tm-st').textContent = ' renombrando…';
          fetch(TM_REN, {method: 'POST',
            headers: {'Content-Type': 'application/x-www-form-urlencoded'},
            body: new URLSearchParams({old: f.tag, new: v})})
            .then(r => r.json()).then(d => {
              if (!d.ok) { tmg('tm-st').textContent = ' ' + (d.error || 'error'); tmPintarFotos(); return; }
              // que el visor abierto lo refleje sin recargar
              try {
                Object.keys(PHOTO_TAGS).forEach(k => {
                  const ts = (PHOTO_TAGS[k] || []).map(t => t === f.tag ? v : t);
                  PHOTO_TAGS[k] = ts.filter((t, i) => ts.indexOf(t) === i);
                });
              } catch (e) {}
              tmg('tm-st').textContent = ' ✓ renombrada en ' + d.cambiadas + ' foto(s)';
              setTimeout(() => tmg('tm-st').textContent = '', 3000);
              tmRecargar();
            }).catch(e => { tmg('tm-st').textContent = ' ' + e; tmPintarFotos(); });
        };
        inp.onkeydown = (e) => {
          if (e.key === 'Enter') { e.preventDefault(); commit(); }
          else if (e.key === 'Escape') cancelar();
        };
        inp.onblur = commit;
      }
      // Renombrar una etiqueta DEL PERFIL (la heredan todas sus fotos)
      function tmEditarPersona(viejo, chipEl) {
        const inp = document.createElement('input');
        inp.className = 'tm-in'; inp.value = viejo; inp.spellcheck = false;
        chipEl.replaceWith(inp); inp.focus(); inp.select();
        let cerrado = false;
        const commit = () => {
          if (cerrado) return;
          cerrado = true;
          const v = inp.value.trim();
          if (!v || v === viejo) { tmPintarPersona(); return; }
          const ts = tmPersona.map(t => t === viejo ? v : t);
          tmPersona = ts.filter((t, i) => ts.indexOf(t) === i);
          tmGuardarPersona();
        };
        inp.onkeydown = (e) => {
          if (e.key === 'Enter') { e.preventDefault(); commit(); }
          else if (e.key === 'Escape') { cerrado = true; tmPintarPersona(); }
        };
        inp.onblur = commit;
      }
      window.tmPintarFotos = tmPintarFotos;
    })();
    </script>

    <div class="meta" style="line-height:1.7">
      <div>🧹 Últ. corrección: <b style="color:var(--tx)">{{ corrected_ago or 'nunca' }}</b></div>
      <div>🔍 Últ. búsqueda: <b style="color:var(--tx)">{{ searched_ago or 'nunca' }}</b></div>
    </div>
  </div>

  <!-- Columna derecha: nombre + ⚙ + etiquetas -->
  <div class="pcard-main">
    <div class="row" style="justify-content:space-between; align-items:center; margin:0">
      <h1 style="margin:0; font-size:19px">{{ name }}</h1>
      <button class="secondary iconbtn" onclick="pcfgAbrir()" title="Configuración de {{ name }}">⚙</button>
    </div>
    {% if bio %}<div class="p-bio">{{ bio }}</div>{% endif %}

    <!-- Modal ⚙: todas las opciones del perfil, en un solo panel scrollable -->
    <div id="pcfg" class="modal-ov" style="display:none" onclick="if(event.target===this) pcfgCerrar()">
      <div class="modal-box" style="width:680px">
        <div class="row" style="justify-content:space-between;margin:0 0 .6rem">
          <b style="font-size:1.05rem">⚙ Configuración de {{ name }}</b>
          <button class="secondary sm" onclick="pcfgCerrar()" title="Cerrar">✕</button>
        </div>
        <div style="max-height:68vh; overflow-y:auto; padding-right:.2rem">

          <form id="pcfg-perfil">
            <b>📝 Bio <span class="hint-ico" title="El párrafo que se ve en su perfil, abajo del nombre: quién es, desde cuándo la conocés…">ⓘ</span></b>
            <textarea name="bio" rows="3" style="width:100%; margin-top:.3rem"
                      placeholder="Quién es {{ name }} (un párrafo corto)…">{{ bio }}</textarea>
            <b style="display:block; margin-top:.7rem">🎙 Voz <span class="hint-ico" title="Cómo habla {{ name }} en los pies de foto que escribe la IA. Si la dejás vacía usa la voz genérica: una cruda y directa, otra tímida…">ⓘ</span></b>
            <textarea name="voice" rows="3" style="width:100%; margin-top:.3rem"
                      placeholder="Ej: «habla con ironía, frases cortas, usa lunfardo»…">{{ voice }}</textarea>
            <div class="row" style="margin:.5rem 0 0; gap:.4rem">
              <button class="primary" type="button" onclick="pcfgGuardar()">💾 Guardar bio y voz</button>
              <span class="meta" id="pcfg-st"></span>
            </div>
          </form>
          <hr class="pcfg-sep">

          {# el vínculo maneja cómo suenan sus pies ({relacion}, {vinculito}), así que
             va acá y no escondido como una etiqueta más del gestor #}
          <form method="post" action="{{ url_for('person_vinculo', name=name) }}" class="row"
                style="margin:0; gap:.4rem">
            <label class="meta">🔗 Vínculo: <span class="hint-ico" title="Define cómo te nombra en los pies: con {relacion} te dice «primo» y con {vinculito} se dice «primita». Necesita tu género en Ajustes para dar vuelta el vínculo.">ⓘ</span></label>
            <select name="vinculo" onchange="this.form.submit()" style="flex:1">
              <option value="">(ninguno)</option>
              {% for v in vinculos_posibles %}
              <option value="{{ v }}" {% if v == vinculo_actual %}selected{% endif %}>{{ v }}</option>
              {% endfor %}
            </select>
          </form>
          <hr class="pcfg-sep">

          <button class="pcfg-item" onclick="document.getElementById('pupload').click()"
                  title="Todas las fotos que elijas se guardan como de {{ name }}, sin analizar caras.">🖼 Subir fotos (todas de {{ name }})…</button>
          <input type="file" id="pupload" accept="image/*" multiple style="display:none" onchange="subirFotos(this)">
          <span class="meta" id="pupload-st" style="display:block;margin:.15rem 0 .2rem"></span>
          <hr class="pcfg-sep">
          {% if display %}
          <button class="pcfg-item" onclick="cfgTagSel()">🏷 Etiquetar con IA…</button>
          {% endif %}
          <button class="pcfg-item" onclick="abrirTagMgr()">🗂 Gestor de etiquetas…</button>
          <a class="pcfg-item" href="{{ url_for('finder_page', name=name) }}">🔍 Buscar sus fotos en otra carpeta</a>
          <a class="pcfg-item" href="{{ url_for('person_export', name=name) }}">📦 Exportar perfil (.zip)</a>
          {% if photos %}
          <form method="post" action="{{ url_for('correct_start', name=name) }}" style="margin:0">
            <button class="pcfg-item" title="Revisa la carpeta y deja a {{ name }} solo con caras verificadas.">🧹 Corregir entrenamiento</button>
          </form>
          <details class="lote" style="margin-top:.2rem">
            <summary class="pcfg-item" style="cursor:pointer">✨ Escribir los posts de todas sus fotos…</summary>
            {% if lote and lote.name == name and lote.status == 'running' %}
              <p class="meta">Corriendo. Podés cerrar esta página: sigue igual.</p>
            {% else %}
            <form method="post" action="{{ url_for('post_lote', name=name) }}">
              <div class="row" style="margin-top:.4rem">
                <select name="que" id="lote-que" onchange="loteEta()">
                  <option value="pie">Solo el pie</option>
                  <option value="comentarios">Solo los comentarios</option>
                  <option value="ambos">Pie y comentarios</option>
                </select>
                <button class="primary" onclick="return confirm('Van ' + document.getElementById('lote-falta').textContent + ' de laburo. ¿Arranco?')">▶ Arrancar</button>
              </div>
              <p class="meta">{{ n_sin_post }} de {{ display_total }} sin hacer · tarda <b id="lote-falta">…</b></p>
            </form>
            {% endif %}
          </details>
          {% endif %}
          <hr class="pcfg-sep">

          <form method="post" action="{{ url_for('person_renombrar', name=name) }}" class="row"
                style="margin:0; gap:.4rem">
            <input class="path" name="nuevo" placeholder="✎ renombrar a…" style="flex:1"
                   value="" autocomplete="off">
            <button class="secondary" title="Mueve su carpeta y migra etiquetas, pies y rostro">Renombrar</button>
          </form>

          {# destructivo: va último y separado, para no apretarlo de paso #}
          <hr class="pcfg-sep">
          <form method="post" action="{{ url_for('known_forget') }}" style="margin:0">
            <input type="hidden" name="name" value="{{ name }}">
            {# OJO: nada de \\n en el confirm. Esto es un string de Python, así que
               \\n se convierte en un salto de línea REAL dentro del onclick y parte
               el string de JS al medio: rompe el atributo y todo lo que sigue. #}
            <button class="pcfg-item" style="color:var(--red)"
                    data-confirm="Eliminar el perfil de {{ name }}: se borran su rostro aprendido, sus etiquetas, sus pies y comentarios. LAS FOTOS NO SE BORRAN, quedan en su carpeta. ¿Seguro?">
              🗑 Eliminar perfil y rostro…
            </button>
          </form>
        </div>
      </div>
    </div>

    {# Las etiquetas de la persona salieron del hero: se manejan en ⚙ → Gestor de
       etiquetas (ahí se editan, renombran y borran, y además están las de fotos). #}

    {% if favs %}
    <!-- Favoritos: carrusel -->
    <div>
      <span class="meta">⭐ Favoritos</span>
      <div class="favwrap">
        <button class="secondary favnav" onclick="favScroll(-1)" title="Anterior">‹</button>
        <div class="favrow" id="favrow">
          {% for r in favs %}
          <img src="{{ url_for('person_thumb', name=name, rel=r) }}" loading="lazy"
               alt="{{ r }}" title="{{ r }}" onclick="favOpen({{ r|tojson }})">
          {% endfor %}
          <a class="favplay" href="{{ url_for('person_favs', name=name) }}" title="Ver la página de favoritas">
            <span style="font-size:18px">▶</span><span>ver<br>todas</span>
          </a>
        </div>
        <button class="secondary favnav" onclick="favScroll(1)" title="Siguiente">›</button>
      </div>
    </div>
    {% endif %}
  </div>
</div>

{% if photos or nsfw_count %}
{% if filter_con %}
<div class="row" style="margin-bottom:.4rem">
  <span class="chip">👥 también aparece <b>{{ filter_con }}</b>
    <a class="x" href="{{ url_for('person_page', name=name, tags=filter_tags or None, per=per, nsfw='1' if show_nsfw else None) }}" title="Quitar este filtro">×</a></span>
</div>
{% endif %}
<form class="row" method="get" action="{{ url_for('person_page', name=name) }}" id="ffiltro">
  {% if show_nsfw %}<input type="hidden" name="nsfw" value="1">{% endif %}
  {% if filter_con %}<input type="hidden" name="con" value="{{ filter_con }}">{% endif %}
  {% if solo_fav %}<input type="hidden" name="fav" value="1">{% endif %}
  <input type="text" id="ffilter" name="tags" class="path" value="{{ filter_tags }}"
         placeholder="🏷 filtrar sus fotos por etiqueta (coma para varias) + Enter…"
         list="tagsug">
  <button class="secondary">Filtrar</button>
  {# filtro por lo que FALTA hacer: para ir tachando pendientes de a tandas #}
  <select name="estado" onchange="this.form.submit()" title="Mostrar solo las que te falta trabajar">
    <option value="">todas</option>
    <option value="sin_etiquetas" {% if estado == 'sin_etiquetas' %}selected{% endif %}>sin etiquetas</option>
    <option value="sin_ia" {% if estado == 'sin_ia' %}selected{% endif %}>sin pasar por la IA</option>
    <option value="ia_vacia" {% if estado == 'ia_vacia' %}selected{% endif %}>la IA no trajo nada (∅)</option>
    <option value="sin_pie" {% if estado == 'sin_pie' %}selected{% endif %}>sin pie</option>
  </select>
  {% if solo_fav %}
  <a class="secondary on" style="padding:.35rem .7rem;border-radius:6px"
     href="{{ url_for('person_page', name=name, tags=filter_tags or None, per=per, con=filter_con or None, nsfw='1' if show_nsfw else None) }}">⭐ Solo favoritos ✕</a>
  {% else %}
  <a class="secondary" style="padding:.35rem .7rem;border-radius:6px"
     href="{{ url_for('person_page', name=name, tags=filter_tags or None, per=per, con=filter_con or None, fav=1, nsfw='1' if show_nsfw else None) }}">☆ Solo favoritos</a>
  {% endif %}
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
                 'con': filter_con or None, 'fav': '1' if solo_fav else None,
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
    <a class="secondary" style="padding:.35rem .7rem;border-radius:6px" href="{{ url_for('person_page', name=name, tags=filter_tags or None, per=per) }}#fotos">🙈 Ver normales</a>
    {% else %}
    <a class="secondary" style="padding:.35rem .7rem;border-radius:6px" href="{{ url_for('person_page', name=name, tags=filter_tags or None, per=per, nsfw=1) }}#fotos">🔞 Solo NSFW ({{ nsfw_count }})</a>
    {% endif %}
  {% endif %}
  {% if display %}
  <button class="secondary" id="tagsel-toggle" onclick="toggleTagSel()">☑ Etiquetar con IA…</button>
  {% endif %}
</div>

{# Panel de selección para el etiquetador automático: aparece con el botón de
   arriba. "Seleccionadas" cuenta solo lo que ves en ESTA página (como el
   etiquetado de a muchas de la Galería general). Sin ninguna tildada, etiqueta
   TODAS las que no tienen etiquetas propias todavía. #}
<div class="panel" id="tagselbar" style="display:none">
  <div class="row">
    <button class="secondary" onclick="tagSelAll(true)">Seleccionar visibles</button>
    <button class="secondary" onclick="tagSelAll(false)">Ninguna</button>
    <span class="meta"><b id="tagselcount">0</b> seleccionada(s) de esta página</span>
  </div>
  <div class="row">
    <button class="primary" id="tagsel-go" onclick="tagLoteStart()">▶ Etiquetar TODAS sin etiquetas propias</button>
    <span class="meta">Mira cada foto y le pone etiquetas sola — mismo motor que el pie. Saltea las que
       ya tienen etiquetas propias, así se puede cortar y retomar.</span>
  </div>
</div>

{# el lote de etiquetado con IA corriendo #}
<div class="panel" id="tagbox" style="display:none">
  <div class="row" style="justify-content:space-between">
    <b>🏷 Etiquetando con IA</b>
    <button class="secondary" onclick="tagLoteStop()" id="tag-stop">■ Parar</button>
  </div>
  <progress id="tag-bar" value="0" max="1" style="width:100%"></progress>
  <p class="meta">
    <b><span id="tag-cur">0</span>/<span id="tag-tot">0</span></b> ·
    lleva <span id="tag-lleva">0 s</span><br>
    <span id="tag-foto" class="fn"></span><br>
    ✓ <span id="tag-hechas">0</span> hechas ·
    ⏭ <span id="tag-salt">0</span> ya tenían ·
    ✕ <span id="tag-fall">0</span> fallaron
  </p>
  <p class="meta" id="tag-err" style="color:var(--red)"></p>
  <ul id="tag-ultimos" class="meta" style="margin:.2rem 0 0;padding-left:1.1rem"></ul>
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
  <div class="shot{% if f.fav %} is-fav{% endif %}" data-tags="{{ f.tags|join(' ') }}" data-rel="{{ f.rel }}">
    <span class="favstar" title="Favorito">⭐</span>
    {% if f.nsfw %}<span class="nsfw-badge" title="{{ f.rel }}">🔞</span>{% endif %}
    {% if f.video %}<span class="vid-badge" title="Video">▶</span>{% endif %}
    {% if not f.video %}
    <label class="shot-selbox" title="Seleccionar para etiquetar con IA">
      <input type="checkbox" class="tagselchk" value="{{ f.rel }}" onchange="tagSelCount()">
    </label>
    {% endif %}
    <div class="shot-thumb" onclick="lbOpen({{ loop.index0 }})" title="{{ f.rel }}">
      <img src="{{ url_for('person_thumb', name=name, rel=f.rel) }}" loading="lazy" alt="{{ f.rel }}">
    </div>
  </div>
  {% endfor %}
</div>
{{ pager() }}

<datalist id="tagsug">{% for t in all_tags %}<option value="{{ t }}">{% endfor %}</datalist>
{# todas las personas conocidas, para etiquetar quién está en la foto (@) #}
<datalist id="knownlist">{% for p in people %}{% if p != name %}<option value="{{ p }}">{% endif %}{% endfor %}</datalist>

<script>
window.KNOWN = {{ people|tojson }};            // todas las personas de la base
window.LB_PHOTOS = [
  {% for f in display %}{url:"{{ url_for('person_photo', name=name, rel=f.rel) }}",name:"{{ f.rel|e }}",rel:{{ f.rel|tojson }},video:{{ f.video|tojson }}},{% endfor %}
];
window.POSTS = {{ posts|tojson }};             // {rel: {caption, comments, people}}
window.COPETES = {{ copetes|tojson }};         // {rel: copete} ya generados
window.VLM_MODELS = {{ vlm_models|tojson }};
window.HAS_COVER = {{ has_thumb|tojson }};
window.COVER_URL_THUMB = "{{ url_for('known_thumb', name=name) }}?v={{ cache_bust }}";
const PERSON = {{ name|tojson }};
let PERSON_TAGS = {{ person_tags|tojson }};
let FAVS = {{ favs_tags|tojson }};             // tus favoritas (persona + globales)
const PHOTO_TAGS = {{ photo_tags|tojson }};   // {rel: [tags propias]}
const PROMPT_TPL = {{ prompt_externo|tojson }};   // plantilla de Ajustes ($etiquetas)
const VINCULO = {{ vinculo|tojson }};             // lo que ES ella (prima), o ""
const RELACION = {{ relacion|tojson }};           // cómo te llama a vos (primo), o ""
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
// El bloque del hero se sacó (ahora se editan en ⚙ → Gestor de etiquetas), así que
// estas funciones no encuentran los elementos: salen sin hacer nada en vez de romper.
function renderPersonTags() {
  const box = document.getElementById('ptags');
  const input = document.getElementById('ptag-input');
  if (!box || !input) return;
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
if (document.getElementById('ptag-input')) {
  tagAC(document.getElementById('ptag-input'), {commit: addPersonTag});
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

// --- el lote de posts (dejarlo toda la noche) ---
// El estimado ANTES de arrancar sale de medidas reales: ~90s el pie, ~60s los
// comentarios. Con 1500 fotos eso da más de 37 horas, y es justo lo que hay que
// saber antes de darle a Arrancar y no después.
const SEGS_POR = {pie: 90, comentarios: 60, ambos: 150};
const SIN_POST = {{ n_sin_post|tojson }};
function humano(s) {
  if (s < 90) return Math.round(s) + ' s';
  if (s < 5400) return Math.round(s / 60) + ' min';
  const h = Math.floor(s / 3600), m = Math.round((s % 3600) / 60);
  return m ? `${h} h ${m} min` : `${h} h`;
}
function loteEta() {
  const sel = document.getElementById('lote-que');
  const el = document.getElementById('lote-falta');
  if (sel && el) el.textContent = '~' + humano(SIN_POST * SEGS_POR[sel.value]);
}
function loteStop() {
  document.getElementById('lote-stop').disabled = true;
  post("{{ url_for('post_lote_stop') }}", {});
}
function loteMirar() {
  fetch("{{ url_for('post_lote_estado') }}").then(r => r.json()).then(d => {
    const box = document.getElementById('lotebox');
    if (d.status !== 'running') {
      box.style.display = 'none';
      // terminó mientras mirabas: recargar para ver los posts nuevos
      if (box.dataset.corria === '1') location.reload();
      return;
    }
    box.dataset.corria = '1';
    box.style.display = '';
    document.getElementById('lote-name').textContent = d.name;
    document.getElementById('lote-bar').value = d.current;
    document.getElementById('lote-bar').max = d.total || 1;
    document.getElementById('lote-cur').textContent = d.current;
    document.getElementById('lote-tot').textContent = d.total;
    document.getElementById('lote-eta').textContent = d.falta;
    document.getElementById('lote-lleva').textContent = d.lleva;
    document.getElementById('lote-foto').textContent = d.photo;
    document.getElementById('lote-hechas').textContent = d.hechas;
    document.getElementById('lote-salt').textContent = d.salteadas;
    document.getElementById('lote-fall').textContent = d.fallaron;
    document.getElementById('lote-err').textContent = d.error || '';
    const ul = document.getElementById('lote-ultimos');
    ul.innerHTML = '';
    (d.ultimos || []).forEach(t => {
      const li = document.createElement('li');
      li.textContent = t;
      ul.appendChild(li);
    });
  }).catch(() => {});
}
loteEta();
loteMirar();
setInterval(loteMirar, 3000);

// --- etiquetador automático: selecciono fotos (o ninguna = todas sin
// etiquetas propias) y la IA les pone etiquetas sola, foto por foto ---
function cfgTagSel() {
  pcfgCerrar();
  if (!tagSelOn) toggleTagSel();
  document.getElementById('tagselbar').scrollIntoView({behavior: 'smooth', block: 'center'});
}
let tagSelOn = false;
function toggleTagSel() {
  tagSelOn = !tagSelOn;
  document.querySelectorAll('.shot-selbox').forEach(b => b.style.display = tagSelOn ? 'block' : 'none');
  document.getElementById('tagselbar').style.display = tagSelOn ? '' : 'none';
  document.getElementById('tagsel-toggle').textContent = tagSelOn ? '✕ Cerrar selección' : '☑ Etiquetar con IA…';
  if (!tagSelOn) tagSelAll(false);
}
function tagSelAll(on) {
  document.querySelectorAll('.tagselchk').forEach(c => { c.checked = on; c.closest('.shot').classList.toggle('tagsel-on', on); });
  tagSelCount();
}
function tagSelCount() {
  const chk = [...document.querySelectorAll('.tagselchk:checked')];
  chk.forEach(c => c.closest('.shot').classList.add('tagsel-on'));
  document.querySelectorAll('.tagselchk:not(:checked)').forEach(c => c.closest('.shot').classList.remove('tagsel-on'));
  document.getElementById('tagselcount').textContent = chk.length;
  document.getElementById('tagsel-go').textContent = chk.length
    ? `▶ Etiquetar ${chk.length} seleccionada(s)`
    : '▶ Etiquetar TODAS sin etiquetas propias';
}
function tagLoteStart() {
  if (!window.VLM_MODELS.length) { alert('Ollama no está corriendo.'); return; }
  const rels = [...document.querySelectorAll('.tagselchk:checked')].map(c => c.value);
  const aviso = rels.length ? `Etiquetar ${rels.length} foto(s) seleccionada(s). ¿Arranco?`
                            : 'Etiquetar TODAS las fotos que todavía no tienen etiquetas propias. ¿Arranco?';
  if (!confirm(aviso)) return;
  post("{{ url_for('tags_lote', name=name) }}", {rels: rels.join('\\n')})
    .then(r => r.json()).then(d => {
      if (!d.ok) { alert(d.error || 'No se pudo arrancar.'); return; }
      toggleTagSel();
      tagLoteMirar();
    });
}
function tagLoteStop() {
  document.getElementById('tag-stop').disabled = true;
  post("{{ url_for('tags_lote_stop') }}", {});
}
function tagLoteMirar() {
  fetch("{{ url_for('tags_lote_estado') }}").then(r => r.json()).then(d => {
    const box = document.getElementById('tagbox');
    if (d.status !== 'running') {
      box.style.display = 'none';
      if (box.dataset.corria === '1') location.reload();
      return;
    }
    box.dataset.corria = '1';
    document.getElementById('tag-stop').disabled = false;
    box.style.display = '';
    document.getElementById('tag-bar').value = d.current;
    document.getElementById('tag-bar').max = d.total || 1;
    document.getElementById('tag-cur').textContent = d.current;
    document.getElementById('tag-tot').textContent = d.total;
    document.getElementById('tag-lleva').textContent = d.lleva;
    document.getElementById('tag-foto').textContent = d.photo;
    document.getElementById('tag-hechas').textContent = d.hechas;
    document.getElementById('tag-salt').textContent = d.salteadas;
    document.getElementById('tag-fall').textContent = d.fallaron;
    document.getElementById('tag-err').textContent = d.error || '';
    const ul = document.getElementById('tag-ultimos');
    ul.innerHTML = '';
    (d.ultimos || []).forEach(t => {
      const li = document.createElement('li');
      li.textContent = t;
      ul.appendChild(li);
    });
  }).catch(() => {});
}
tagLoteMirar();
setInterval(tagLoteMirar, 3000);

// --- modal de configuración de la persona (⚙) ---
function pcfgAbrir() {
  const m = document.getElementById('pcfg');
  if (m) m.style.display = 'flex';   // flex, no block: .modal-ov centra con flexbox
}
function pcfgCerrar() {
  const m = document.getElementById('pcfg');
  if (m) m.style.display = 'none';
}
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') {
    const m = document.getElementById('pcfg');
    if (m && m.style.display !== 'none') pcfgCerrar();
  }
});
// guardar bio + voz del modal (AJAX: actualiza el perfil sin recargar)
const PCFG_URL = "{{ url_for('person_perfil', name=name) }}";
function pcfgGuardar() {
  const st = document.getElementById('pcfg-st');
  const fd = new URLSearchParams();
  fd.set('bio', document.querySelector('#pcfg-perfil textarea[name=bio]').value);
  fd.set('voice', document.querySelector('#pcfg-perfil textarea[name=voice]').value);
  st.textContent = ' guardando…';
  fetch(PCFG_URL, {method: 'POST',
    headers: {'Content-Type': 'application/x-www-form-urlencoded'}, body: fd})
    .then(r => r.json()).then(d => {
      if (!d.ok) { st.textContent = ' ✗ ' + d.error; return; }
      st.textContent = ' ✓ guardado';
      setTimeout(() => st.textContent = '', 2000);
      // reflejar la bio nueva bajo el nombre sin recargar
      const bio = document.querySelector('.p-bio');
      if (d.bio) {
        if (bio) bio.textContent = d.bio;
        else {
          const el = document.createElement('div');
          el.className = 'p-bio'; el.textContent = d.bio;
          document.querySelector('.pcard-main .row').insertAdjacentElement('afterend', el);
        }
      } else if (bio) bio.remove();
    }).catch(e => { st.textContent = ' ' + e; });
}

// --- subir fotos directo al perfil (todas quedan como de esta persona) ---
const SUBIR_URL = "{{ url_for('person_subir', name=name) }}";
function subirFotos(input) {
  const files = input.files; if (!files || !files.length) return;
  const st = document.getElementById('pupload-st');
  st.textContent = ' subiendo ' + files.length + ' foto(s)…';
  const fd = new FormData();
  for (const f of files) fd.append('fotos', f);
  fetch(SUBIR_URL, {method: 'POST', body: fd}).then(r => r.json()).then(d => {
    input.value = '';
    if (!d.ok) { st.textContent = ' ✗ ' + d.error; return; }
    st.textContent = ' ✓ ' + d.copiadas + ' subidas' + (d.dup ? ', ' + d.dup + ' repetidas' : '');
    if (d.copiadas) setTimeout(() => location.reload(), 1200);   // que aparezcan
  }).catch(e => { st.textContent = ' ' + e; input.value = ''; });
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
<style>
  /* video en el visor: panel completo igual que en las fotos. Los botones de IA
     (pie/comentarios/sugerir) no funcionan sobre video, pero el resto (favorito,
     mover, NSFW, navegación, etiquetas a mano) sí. */
  #lb-vid { background:#000; }
</style>
<div id="lb" class="lb-post">
  <button class="lb-btn" id="lb-prev" onclick="lbStep(-1)" title="Anterior">‹</button>
  <button class="lb-btn" id="lb-next" onclick="lbStep(1)" title="Siguiente">›</button>
  <div class="post">
    <div class="stage">
      <img id="lb-img" src="" alt="">
      <video id="lb-vid" controls playsinline preload="metadata"
             style="display:none; max-width:100%; max-height:100%; outline:none"></video>
      <button class="secondary sm stage-fab stage-fab-l" onclick="lbCover(this)"
              title="Usar esta foto como portada de la galería">🖼 Portada</button>
      <button class="secondary sm stage-fab stage-fab-r" onclick="openPeopleModal()"
              title="Etiquetar quién más está en la foto">👥 Personas ▾</button>
      <!-- "No es X": mover la foto a otra persona o sacarla; sobre la foto, abajo -->
      <form method="post" action="{{ url_for('person_move_photo', name=name) }}"
            id="lb-move-form" class="stage-fab stage-fab-bl" style="margin:0">
        <input type="hidden" name="rel" id="lb-move-rel" value="">
        <select name="to" class="secondary sm" style="font-size:.78rem; max-width:210px"
                onchange="if (this.value) this.form.submit()">
          <option value="">🚚 No es {{ name }}… mover a</option>
          <option value="__review__">🗂 Sacar a _revisar</option>
          {% for p in people %}{% if p != name %}<option value="{{ p }}">👤 {{ p }}</option>{% endif %}{% endfor %}
        </select>
      </form>
    </div>
    <aside class="side">
      <header class="phead">
        <img id="lb-av" class="av" src="" alt="">
        <div style="flex:1">
          <b id="lb-who"></b>
          <div class="meta" id="lb-fn"></div>
        </div>
        <button id="lb-fav" class="favbtn" onclick="toggleFav()" title="Favorito">☆</button>
      </header>

      <div class="pbody">
        <div class="pcap pcard-sub" id="lb-cap-box">
          <div class="caprow">
            <div class="capmain">
              <p id="lb-cap" class="empty">Todavía no tiene pie de foto.</p>
              <div id="lb-cap-sk" style="display:none">
                <div class="sk sk-linea" style="width:96%"></div>
                <div class="sk sk-linea" style="width:88%"></div>
                <div class="sk sk-linea" style="width:60%"></div>
              </div>
              <span class="meta" id="lb-cap-st"></span>
            </div>
            <button class="primary capfab" id="lb-cap-btn" onclick="genCaption()" title="Escribir el pie con IA">✨</button>
            <button class="secondary capfab" id="lb-cap-edit" onclick="editCap()" title="Reescribirlo a tu manera">✏️</button>
            <button class="secondary capfab" id="lb-cap-molde" onclick="abrirMoldeFoto()" title="Armar un molde de pie con las etiquetas de esta foto">✍</button>
          </div>
          <textarea id="lb-cap-in" class="capin" style="display:none"
                    placeholder="escribilo a tu manera… (Ctrl+Enter para guardar)"></textarea>
        </div>

        <!-- personas etiquetadas: solo aparece si hay alguien (se agrega con el
             botón "👥 Personas" de arriba de la foto) -->
        <div class="ppeople" id="lb-people-box" style="display:none">
          <div class="chips" id="lb-people">
            <input class="taginput" id="lb-people-in" list="knownlist" autocomplete="off"
                   placeholder="@ agregar a otra persona…">
          </div>
        </div>

        <div class="pcoms pcard-sub">
          <div class="row" style="justify-content:space-between;margin:0 0 .3rem">
            <span class="meta">💬 Comentarios</span>
            <span style="display:flex;gap:.35rem">
              <button class="primary capfab" id="lb-com-btn" onclick="genComments()" title="Generar comentarios con IA">✨</button>
              <button class="secondary capfab" id="lb-com-x" onclick="clearComments()" title="Borrarlos y empezar de nuevo">🗑</button>
            </span>
          </div>
          <div id="lb-coms"></div>
          <div id="lb-com-sk" style="display:none">
            {% for w in [70, 85, 62, 78, 55, 80] %}
            <div style="padding:.35rem 0;border-bottom:1px solid var(--line)">
              <span class="sk" style="display:inline-block;height:.7rem;width:3.5rem"></span>
              <span class="sk" style="display:inline-block;height:.7rem;width:{{ w }}%;margin-left:.3rem"></span>
            </div>
            {% endfor %}
          </div>
          <span class="meta" id="lb-com-st"></span>
        </div>

        <!-- etiquetas: al fondo y discreto. Muestra las primeras 3 + "ver más". -->
        <div class="ptags-min">
          <div id="lb-tag-preview" class="lb-tagprev"></div>
          <button class="linklike" id="lb-tag-toggle" onclick="openTagModal()">🏷 Editar / Sugerir con IA</button>
        </div>

        <!-- atajo a la columna de la derecha: abre el Centro de IA en su propio
             modal (Visión → Etiquetas → Pie → Comentarios → Copete) -->
        <button class="primary cat-button" type="button" onclick="openIA()">✨ Crear contenido</button>
      </div>

      <!-- acciones de ESTA foto (antes estaban en cada tarjeta de la grilla).
           El "No es X… mover" se movió arriba, sobre la foto. -->
      <div class="pact">
        <button type="button" class="secondary sm" id="lb-nsfw-btn" onclick="marcarNsfw()"
                title="Marcar como NSFW: la mueve a _nsfw y corre el detector de acto 🎯">🔞 NSFW</button>
      </div>

      <div class="pfoot">
        <select id="lb-model" title="Con qué modelo escribe la IA">
          {% for m in vlm_models %}<option value="{{ m }}">{{ m }}</option>{% endfor %}
        </select>
        <span class="meta" id="lb-pos"></span>
      </div>
    </aside>
  </div>
  <div id="lb-people-modal" class="modal-ov" style="display:none" onclick="if(event.target===this) closePeopleModal()">
    <div class="modal-box">
      <div class="row" style="justify-content:space-between;margin:0">
        <b>🏷️ Etiquetar persona</b>
        <button class="secondary sm" onclick="closePeopleModal()" title="Cerrar">✕</button>
      </div>
      <input class="path" id="pmodal-search" placeholder="🔍 Buscar por nombre…" autocomplete="off"
             oninput="pmodalFilter()" style="width:100%;margin:.5rem 0">
      <div id="pmodal-list" class="pmodal-list"></div>
    </div>
  </div>

  <!-- modal: armar molde de pie con las etiquetas de ESTA foto -->
  <div id="lb-molde-modal" class="modal-ov" style="display:none" onclick="if(event.target===this) cerrarMoldeFoto()">
    <div class="modal-box" style="width:600px">
      <div class="row" style="justify-content:space-between;margin:0 0 .4rem">
        <b style="font-size:1.05rem">✍ Molde de pie de esta foto</b>
        <button class="secondary sm" onclick="cerrarMoldeFoto()" title="Cerrar">✕</button>
      </div>
      <textarea id="lb-molde-json" class="textos" spellcheck="false" rows="12" style="width:100%"></textarea>
      <div class="row" style="margin-top:.4rem">
        <button class="secondary" onclick="copiarMoldeFoto()">📋 Copiar JSON</button>
        <button class="secondary" id="lb-molde-prompt" onclick="copiarPromptFoto()" title="Copia el prompt de Ajustes con las etiquetas de esta foto">🧩 Copiar prompt</button>
        <button class="primary" onclick="guardarMoldeFoto()">💾 Guardar y traer el pie</button>
        <span class="meta" id="lb-molde-st"></span>
      </div>
    </div>
  </div>

  <!-- modal grande de etiquetas: aplicadas + agregar a mano + sugerir con IA -->
  <div id="lb-tag-modal" class="modal-ov" style="display:none" onclick="if(event.target===this) closeTagModal()">
    <div class="modal-box tagmodal">
      <div class="row" style="justify-content:space-between;margin:0 0 .3rem">
        <b style="font-size:1.05rem">🏷️ Etiquetas de la foto</b>
        <button class="secondary sm" onclick="closeTagModal()" title="Cerrar">✕</button>
      </div>
      <div class="tagm-cols">
        <!-- izquierda: etiquetas y sugerencias -->
        <div class="tagm-body">
          <span class="meta">Heredadas de {{ name }} <span style="opacity:.6">· valen para todas sus fotos</span></span>
          <div class="chips" id="lb-inh"></div>

          <span class="meta" style="display:block;margin-top:.7rem">Etiquetas de ESTA foto:</span>
          <div class="chips" id="lb-own">
            <input class="taginput" id="lb-tin" placeholder="escribí una y Enter…">
          </div>

          <!-- tus favoritas: un clic y se agregan (para no tipear siempre lo mismo) -->
          <span class="meta" style="display:block;margin-top:.7rem">⭐ Tus favoritas
            <span style="opacity:.6">· clic para agregarla</span></span>
          <div class="chips" id="lb-favs"></div>

          <div class="row" style="margin-top:.9rem; align-items:center; gap:.5rem">
            <span class="meta">Modelo:</span>
            <select id="lb-tagmodel" title="Con qué modelo mira la foto para etiquetar"
                    style="font-size:.82rem; max-width:60%">
              {% for m in vlm_models %}
              <option value="{{ m }}" {% if m == tag_modelo %}selected{% endif %}>{% if m in vlm_crudos %}🔞 {% else %}👁 {% endif %}{{ m }}</option>
              {% endfor %}
            </select>
          </div>
          <span class="meta" style="display:block;margin:.25rem 0 .6rem">
            🔞 = sin censura (no rebota lo explícito, pero <b>ve peor</b>) ·
            👁 = ve mejor pero puede suavizar lo crudo</span>
          <div class="row">
            <button class="primary act" id="lb-sug" onclick="suggestTags()">🔮 Sugerir con IA</button>
            <span class="meta" id="lb-sug-st"></span>
          </div>
          <span class="meta" id="lb-sug-count" style="display:block;margin-top:.5rem"></span>
          <div class="suglist" id="lb-sugtags"></div>
          <div id="lb-sug-sk" style="display:none;margin-top:.4rem">
            {% for w in [92, 78, 85, 70, 88, 62] %}<div class="sk sk-linea" style="width:{{ w }}%;height:1.6rem"></div>{% endfor %}
          </div>
        </div>
        <!-- derecha: la foto, para chequear cada sugerencia contra lo que se ve -->
        <div class="tagm-photo"><img id="tagm-photo" src="" alt="la foto"></div>
      </div>
    </div>
  </div>
</div>

<!-- modal: Centro de IA — acordeón de pasos encadenados para el contenido de
     la foto. Se abre con "✨ Crear contenido" (abajo de pie/comentarios):
     1) Visión = la IA mira y anota qué hay. Con eso 2) Etiquetas, 3) Pie,
     4) Comentarios. 5) Copete = SOLO titular para la portada: no va como texto
     del pie ni en el cuerpo de la foto. -->
<div id="lb-ia-modal" class="modal-ov" style="display:none"
     onclick="if(event.target===this) closeIA()">
  <div class="modal-box iamodal">
    <div class="row" style="justify-content:space-between;margin:0 0 .4rem">
      <b style="font-size:1.05rem">🧠 Centro de IA ·
        <span class="meta" id="ia-who"></span></b>
      <button class="secondary sm" onclick="closeIA()" title="Cerrar">✕</button>
    </div>
    <div class="pia" id="lb-pia">
      <!-- 1 · Visión -->
      <details class="pia-step" id="pia-vision" open>
        <summary><span>1 · 🔎 Visión</span><span class="meta">qué hay en la foto</span></summary>
        <div class="pia-body">
          <p id="ia-vision" class="empty">La IA todavía no miró la foto: tocar ✨.</p>
          <div id="ia-vision-sk" style="display:none">
            <div class="sk sk-linea" style="width:96%"></div>
            <div class="sk sk-linea" style="width:80%"></div>
          </div>
          <div class="pia-actions">
            <button class="primary capfab" id="ia-vision-btn" onclick="genVisionIA()"
                    title="Mirar la foto y anotar qué hay (paso 1 del Centro de IA)">✨</button>
            <span class="meta" id="ia-vision-st"></span>
          </div>
        </div>
      </details>
      <!-- 2 · Etiquetas -->
      <details class="pia-step" id="pia-tags">
        <summary><span>2 · 🏷 Etiquetas</span></summary>
        <div class="pia-body pia-tags">
          <div id="ia-tag-preview" class="lb-tagprev"></div>
          <div class="pia-actions">
            <button class="primary act" id="ia-sug" onclick="sugIA()">🔮 Sugerir con IA</button>
            <span class="meta" id="ia-sug-st"></span>
          </div>
          <span class="meta" id="ia-sug-count" style="display:block;margin-top:0"></span>
          <div class="suglist" id="ia-sugtags"></div>
          <div id="ia-sug-sk" style="display:none;margin-top:.4rem">
            {% for w in [92, 78, 85, 70, 88, 62] %}<div class="sk sk-linea" style="width:{{ w }}%;height:1.6rem"></div>{% endfor %}
          </div>
          <button class="linklike" id="ia-tag-toggle" onclick="openTagModal()"
                  title="Etiquetas heredadas del perfil, tus favoritas y gestor completo">🛠 Más opciones (gestor completo)</button>
        </div>
      </details>
      <!-- 3 · Pie -->
      <details class="pia-step" id="pia-pie">
        <summary><span>3 · ✍ Pie de foto</span></summary>
        <div class="pia-body">
          <p id="ia-cap" class="empty">Todavía no tiene pie de foto.</p>
          <div id="ia-cap-sk" style="display:none">
            <div class="sk sk-linea" style="width:96%"></div>
            <div class="sk sk-linea" style="width:88%"></div>
            <div class="sk sk-linea" style="width:60%"></div>
          </div>
          <div class="pia-actions">
            <button class="primary capfab" id="ia-cap-btn" onclick="genCaptionIA()"
                    title="Escribir el pie con IA">✨</button>
            <span class="meta" id="ia-cap-st"></span>
          </div>
        </div>
      </details>
      <!-- 4 · Comentarios -->
      <details class="pia-step" id="pia-coms">
        <summary><span>4 · 💬 Comentarios</span></summary>
        <div class="pia-body">
          <div id="ia-coms"></div>
          <div id="ia-com-sk" style="display:none">
            {% for w in [70, 85, 62, 78, 55, 80] %}
            <div style="padding:.35rem 0;border-bottom:1px solid var(--line)">
              <span class="sk" style="display:inline-block;height:.7rem;width:3.5rem"></span>
              <span class="sk" style="display:inline-block;height:.7rem;width:{{ w }}%;margin-left:.3rem"></span>
            </div>
            {% endfor %}
          </div>
          <div class="pia-actions">
            <button class="primary capfab" id="ia-com-btn" onclick="genCommentsIA()"
                    title="Generar comentarios con IA">✨</button>
            <span class="meta" id="ia-com-st"></span>
          </div>
        </div>
      </details>
      <!-- 5 · Copete (solo portada) -->
      <details class="pia-step" id="pia-cop">
        <summary><span>5 · 📰 Copete</span><span class="meta">solo portada</span></summary>
        <div class="pia-body">
          <p id="ia-cop" class="empty">Todavía no tiene copete.</p>
          <div id="ia-cop-sk" style="display:none">
            <div class="sk sk-linea" style="width:96%"></div>
            <div class="sk sk-linea" style="width:70%"></div>
          </div>
          <div class="pia-actions">
            <button class="primary capfab" id="ia-cop-btn" onclick="genCopeteIA()"
                    title="Inventar un titular de noticia con IA (mira la foto)">📰</button>
            <button class="secondary capfab" id="ia-cop-home" onclick="usarCopeteHomeIA()"
                    title="Poner este copete como destacado de la portada">🏠</button>
            <span class="meta" id="ia-cop-st"></span>
          </div>
        </div>
      </details>
    </div>
    <span class="meta" style="display:block;margin-top:.6rem">Con el modelo
      <b id="ia-model-note"></b> · lo que ya escribiste (<span class="meta">pie, comentarios, etiquetas</span>) queda igual.
    </span>
  </div>
</div>

<script>
(function() {
  const P = window.LB_PHOTOS || [], POSTS = window.POSTS || {};
  const lb = document.getElementById('lb'), img = document.getElementById('lb-img');
  const $ = id => document.getElementById(id);
  let idx = 0, scale = 1, tx = 0, ty = 0, drag = false, sx = 0, sy = 0;
  let ownExpanded = false;     // ¿mostrar TODAS las etiquetas propias o colapsadas?
  const OWN_LIMIT = 3;         // cuántas propias se ven antes del "mostrar todas"
                              // (no importan a la vista; están igual para la IA)
  const CAP_URL = "{{ url_for('post_caption', name=name) }}";
  const VISION_URL = "{{ url_for('post_vision', name=name) }}";
  const COPETE_URL = "{{ url_for('post_copete', name=name) }}";
  const CAPED_URL = "{{ url_for('post_caption_edit', name=name) }}";
  const COM_URL = "{{ url_for('post_comments', name=name) }}";
  const COMX_URL = "{{ url_for('post_comments_clear', name=name) }}";
  const SUG_URL = "{{ url_for('tags_suggest', name=name) }}";
  const SUG_STREAM_URL = "{{ url_for('tags_suggest_stream', name=name) }}";
  const PEOPLE_URL = "{{ url_for('post_people', name=name) }}";
  const COMBO_URL = "{{ url_for('moldes_combo') }}";
  const IMPORT_URL = "{{ url_for('moldes_import') }}";
  const DESDE_FOTO_URL = "{{ url_for('moldes_desde_foto', name=name) }}";

  const rel = () => P[idx].rel;
  const post_ = (url, body) => fetch(url, {method:'POST',
      headers:{'Content-Type':'application/x-www-form-urlencoded'},
      body: new URLSearchParams(body)}).then(r => r.json());
  const esc = s => String(s).replace(/[<>&]/g, c => ({'<':'&lt;','>':'&gt;','&':'&amp;'}[c]));
  function slot(r) { return POSTS[r] || (POSTS[r] = {caption:null, comments:[], people:[], vision:null, copete:null}); }
  function copeteDe(r) { return (window.COPETES[r] || '').trim() || null; }

  // --- acordeón del Centro de IA: estado por foto ---
  // Al abrir una foto nueva se despliegan los pasos que ya tienen contenido; al
  // regenerar en la MISMA foto no se toca lo que el usuario dejó abierto/cerrado
  // (antes se forzaba "Visión abierto" y se reabría solo al crear el pie).
  let piaLast = null;        // para qué foto es el estado ({rel} cuando se abrió)
  function marcaPia() {
    const r = rel();
    if (piaLast === r) return;                 // misma foto: respetá lo que dejó
    piaLast = r;
    const d = slot(r), cop = d.copete || copeteDe(r);
    const abrir = id => { const el = $(id); if (el) el.open = true; };
    abrir('pia-vision');                        // Visión siempre: es la entrada
    if ((PHOTO_TAGS[r] || []).length > 0) abrir('pia-tags');
    if (d.caption) abrir('pia-pie');
    if ((d.comments || []).length > 0) abrir('pia-coms');
    if (cop) abrir('pia-cop');
  }

  // --- zoom y arrastre (igual que el visor pelado) ---
  function apply() { img.style.transform = `translate(${tx}px,${ty}px) scale(${scale})`; }
  function reset() { scale = 1; tx = 0; ty = 0; apply(); }

  function render() {
    // Si pasaste de foto mientras la IA escribía, el esqueleto no puede quedar
    // colgado en la foto nueva. El trabajo sigue en segundo plano y se guarda
    // igual en la foto que lo pidió (por eso el `r === rel()` de más abajo).
    Object.keys(TRABAJOS).forEach(k => trabajando(k, false));
    const d = slot(rel());
    $('lb-cap').textContent = d.caption || 'Todavía no tiene pie de foto.';
    $('lb-cap').className = d.caption ? '' : 'empty';
    // el botón queda siempre como ✨ (compacto): lo que cambia es el tooltip
    $('lb-cap-btn').title = d.caption ? 'Reescribir el pie con IA' : 'Escribir el pie con IA';
    $('lb-cap-edit').style.display = d.caption ? '' : 'none';
    cancelCap();
    // modal Centro de IA: visión, pie, comentarios y copete se repintan con los
    // mismos datos (lo que se genera vale para la columna y para el modal)
    $('ia-vision').textContent = d.vision || 'La IA todavía no miró la foto: tocar ✨.';
    $('ia-vision').className = d.vision ? '' : 'empty';
    $('ia-vision-st').textContent = '';
    $('ia-vision-btn').title = d.vision ? 'Volver a mirar la foto' : 'Mirar la foto y anotar qué hay (paso 1 del Centro de IA)';
    $('ia-cap').textContent = d.caption || 'Todavía no tiene pie de foto.';
    $('ia-cap').className = d.caption ? '' : 'empty';
    $('ia-cap-st').textContent = '';
    // copete: el guardado de la foto, o el que se acaba de pedir
    const cop = d.copete || copeteDe(rel());
    $('ia-cop').textContent = cop || 'Todavía no tiene copete.';
    $('ia-cop').className = cop ? '' : 'empty';
    $('ia-cop-home').style.visibility = cop ? '' : 'hidden';
    $('ia-cop-st').textContent = '';
    const ib = $('ia-coms');
    ib.innerHTML = d.comments.length ? '' : '<span class="meta empty">Sin comentarios.</span>';
    d.comments.forEach(c => {
      const el = document.createElement('div'); el.className = 'com';
      el.innerHTML = '<b>@' + esc(c.user) + '</b> ' + esc(c.text);
      ib.appendChild(el);
    });
    $('ia-com-st').textContent = '';
    if ($('ia-who')) $('ia-who').textContent = PERSON;
    if ($('ia-model-note')) $('ia-model-note').textContent = $('lb-model').value;
    const box = $('lb-coms');
    box.innerHTML = d.comments.length ? '' : '<span class="meta empty">Sin comentarios.</span>';
    d.comments.forEach(c => {
      const el = document.createElement('div');
      el.className = 'com';
      el.innerHTML = '<b>@' + esc(c.user) + '</b> ' + esc(c.text);
      box.appendChild(el);
    });
    $('lb-com-btn').title = d.comments.length ? 'Generar más comentarios con IA' : 'Generar comentarios con IA';
    $('lb-com-x').style.display = d.comments.length ? '' : 'none';
    $('lb-cap-st').textContent = ''; $('lb-com-st').textContent = '';
    $('lb-sug-st').textContent = ''; $('lb-sugtags').innerHTML = '';
    if ($('lb-sug-count')) $('lb-sug-count').textContent = '';
    $('ia-sug-st').textContent = ''; $('ia-sugtags').innerHTML = '';
    if ($('ia-sug-count')) $('ia-sug-count').textContent = '';
    ownExpanded = false;                         // etiquetas propias colapsadas
    _tagPrevExp = false;                          // preview de etiquetas colapsado
    sugestLB.reset(); sugestIA.reset();           // corta cualquier auto-continuo en curso
    renderPeople();
    renderFav();
    renderTags();
    $('lb-move-rel').value = rel();   // el "mover a" siempre apunta a ESTA foto
    // el botón NSFW no va si la foto YA es nsfw
    $('lb-nsfw-btn').style.display = rel().split('/').includes('_nsfw') ? 'none' : '';
    // el acordeón del Centro de IA: al CAMBIAR de foto abre los pasos que tienen
    // contenido (Visión siempre, es la entrada del flujo). Al regenerar en la
    // MISMA foto respeta lo que el usuario dejó abierto/cerrado.
    marcaPia();
  }

  // portada desde el visor: abre el EDITOR con esta foto (recorte a mano) en vez
  // de ponerla directo. Si el editor no está (otra página), cae al modo viejo.
  window.lbCover = function(btn) {
    if (window.coverEditar) coverEditar(rel(), P[idx].url);
    else setCover(btn, rel());
  };
  // marcar ESTA foto como NSFW: la mueve a _nsfw y ahí corre el detector de acto.
  // ASÍNCRONO: actualiza en el lugar, NO recarga (no te cierra el visor).
  window.marcarNsfw = function() {
    const r = rel();
    if (!confirm('¿Marcar esta foto como NSFW? Se mueve a _nsfw y ahí corre el detector de acto (🎯).')) return;
    post_("{{ url_for('post_mark_nsfw', name=name) }}", {rel: r}).then(d => {
      if (!d.ok) { alert(d.error || 'No se pudo marcar.'); return; }
      // muevo el estado en memoria a la nueva ruta, sin recargar
      if (POSTS[r]) { POSTS[d.new_rel] = POSTS[r]; delete POSTS[r]; }
      if (PHOTO_TAGS[r]) { PHOTO_TAGS[d.new_rel] = PHOTO_TAGS[r]; delete PHOTO_TAGS[r]; }
      P[idx].rel = d.new_rel; P[idx].url = d.new_url;   // el visor sigue en ESTA foto
      $('lb-move-rel').value = d.new_rel;
      $('lb-nsfw-btn').style.display = 'none';           // ya es nsfw
      // saco su tarjeta de la grilla de atrás (ya no va en la vista no-nsfw)
      const card = document.querySelector('.shot[data-rel="' + (window.CSS && CSS.escape ? CSS.escape(r) : r) + '"]');
      if (card) card.remove();
      const pos = $('lb-pos'), antes = pos.textContent;
      pos.textContent = '✓ marcada NSFW';
      setTimeout(() => { if (rel() === d.new_rel) pos.textContent = antes; }, 2500);
    });
  };
  // carrusel de favoritos de la card
  window.favScroll = function(dir) {
    const t = document.getElementById('favrow');
    if (t) t.scrollBy({left: dir * 270, behavior: 'smooth'});
  };
  window.favOpen = function(r) {
    const i = P.findIndex(p => p.rel === r);
    if (i >= 0) lbOpen(i);
    else location = '?fav=1';   // está en otra página del paginado
  };

  // --- favorito (⭐) ---
  const FAV_URL = "{{ url_for('post_fav', name=name) }}";
  function renderFav() {
    const on = !!slot(rel()).fav;
    const b = $('lb-fav');
    // ★ y no el emoji ⭐: el emoji es más ancho que el botón de 34px y se
    // desbordaba (se veía reventado). El color ámbar ya marca que está activa.
    b.textContent = on ? '★' : '☆';
    b.classList.toggle('on', on);
  }
  function toggleFav() {
    const r = rel(), nuevo = !slot(r).fav;
    slot(r).fav = nuevo; renderFav();
    post_(FAV_URL, {rel: r, fav: nuevo ? '1' : ''}).then(d => {
      if (d.ok) { slot(r).fav = d.fav; renderFav(); syncGridFav(r, d.fav); }
    });
  }
  function syncGridFav(r, on) {   // marca la tarjeta de atrás
    const card = document.querySelector(`.shot[data-rel="${CSS.escape(r)}"]`);
    if (card) card.classList.toggle('is-fav', on);
  }
  window.toggleFav = toggleFav;

  // --- personas en la foto (@) ---
  function people() { return (slot(rel()).people || []).slice(); }
  function savePeople(lista) {
    const r = rel();
    slot(r).people = lista;
    post_(PEOPLE_URL, {rel: r, people: lista.join(',')}).then(d => {
      if (d.ok) { slot(r).people = d.people; renderPeople(); }
    });
  }
  function renderPeople() {
    const box = $('lb-people'), input = $('lb-people-in');
    const gente = people();
    // la sección solo aparece si hay alguien etiquetado (se agrega con el botón
    // "👥 Personas" de arriba de la foto)
    $('lb-people-box').style.display = gente.length ? '' : 'none';
    box.querySelectorAll('.chip').forEach(c => c.remove());
    gente.forEach(p => {
      const c = document.createElement('span');
      c.className = 'chip';
      // el nombre es un link: filtra la galería a las fotos donde también está
      const url = '{{ url_for("person_page", name=name) }}?con=' + encodeURIComponent(p);
      c.innerHTML = '<a href="' + url + '" title="Ver las fotos donde también está ' +
        esc(p) + '" style="color:inherit">@' + esc(p) + '</a> <span class="x">×</span>';
      c.querySelector('.x').onclick = () => savePeople(people().filter(x => x !== p));
      box.insertBefore(c, input);
    });
  }
  $('lb-people-in').addEventListener('keydown', e => {
    if (e.key === 'Enter') {
      e.preventDefault();
      const v = e.target.value.trim().replace(/^@/, '');
      // solo si existe en la base y no está ya puesto
      if (v && window.KNOWN.includes(v) && v !== PERSON && !people().includes(v))
        savePeople(people().concat([v]));
      e.target.value = '';
    } else if (e.key === 'Escape') { e.stopPropagation(); }
  });

  // --- modal "🏷️ Etiquetar persona": buscar y agregar con avatar ---
  function pmodalFilter() {
    const q = $('pmodal-search').value.trim().toLowerCase();
    const ya = people();
    const cands = (window.KNOWN || [])
      .filter(p => p !== PERSON)
      .filter(p => !q || p.toLowerCase().includes(q))
      .sort();
    const list = $('pmodal-list');
    list.innerHTML = '';
    if (!cands.length) {
      list.innerHTML = '<p class="meta" style="padding:.5rem">Nadie más en la base.</p>';
      return;
    }
    cands.forEach(p => {
      const puesta = ya.includes(p);
      const it = document.createElement('div');
      it.className = 'pmodal-item' + (puesta ? ' added' : '');
      const av = document.createElement('img');
      av.src = '/known_thumb/' + encodeURIComponent(p);
      av.alt = '';
      av.onerror = () => { av.style.visibility = 'hidden'; };
      const nm = document.createElement('span');
      nm.style.flex = '1';
      nm.textContent = p;
      const st = document.createElement('span');
      st.className = 'meta';
      st.textContent = puesta ? '✓ ya está' : '+ agregar';
      it.appendChild(av); it.appendChild(nm); it.appendChild(st);
      if (!puesta) it.onclick = () => { savePeople(people().concat([p])); pmodalFilter(); };
      list.appendChild(it);
    });
  }
  function openPeopleModal() {
    $('pmodal-search').value = '';
    pmodalFilter();
    $('lb-people-modal').style.display = 'flex';
    $('pmodal-search').focus();
  }
  function closePeopleModal() { $('lb-people-modal').style.display = 'none'; }
  window.openPeopleModal = openPeopleModal;
  window.closePeopleModal = closePeopleModal;
  window.pmodalFilter = pmodalFilter;

  const vid = $('lb-vid');
  function show(i) {
    idx = (i + P.length) % P.length;
    reset();
    const it = P[idx];
    // video vs imagen: se muestra <video> y se esconde la columna de acciones
    // (pie/comentarios/etiquetas son sobre imágenes, no aplican al video)
    lb.classList.toggle('is-video', !!it.video);
    if (it.video) {
      img.style.display = 'none'; img.src = '';
      vid.src = it.url; vid.style.display = '';
    } else {
      vid.pause(); vid.removeAttribute('src'); vid.load(); vid.style.display = 'none';
      img.style.display = ''; img.src = it.url;
    }
    $('lb-fn').textContent = it.name;
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
  window.lbClose = function() {
    lb.style.display = 'none'; img.src = '';
    vid.pause(); vid.removeAttribute('src'); vid.load();
  };
  window.lbStep = function(d) { show(idx + d); };

  // --- estado "la IA está trabajando" ---
  // Tarda entre 30s y 2 minutos. Sin señal clara, el usuario aprieta el botón,
  // ve el texto viejo quieto y no sabe si lo apretó. Así que: se esconde lo
  // viejo, aparece el esqueleto, y el botón dice qué está haciendo y cuánto.
  const RELOJ = ['⠋','⠙','⠹','⠸','⠼','⠴','⠦','⠧','⠇','⠏'];
  const TRABAJOS = {
    cap: {btn: 'lb-cap-btn', st: 'lb-cap-st', sk: 'lb-cap-sk',
          ver: ['lb-cap', 'lb-cap-edit'], texto: 'escribiendo el pie…', segundos: 90},
    com: {btn: 'lb-com-btn', st: 'lb-com-st', sk: 'lb-com-sk',
          ver: ['lb-coms', 'lb-com-x'], texto: 'escribiendo…', segundos: 60},
    // sug NO esconde lb-sugtags: las etiquetas caen en vivo ahí mientras trabaja
    sug: {btn: 'lb-sug', st: 'lb-sug-st', sk: 'lb-sug-sk',
          ver: [], texto: 'mirando la foto…', segundos: 35},
    copIA: {btn: 'ia-cop-btn', st: 'ia-cop-st', sk: 'ia-cop-sk',
          ver: ['ia-cop', 'ia-cop-home'], texto: 'inventando el titular…', segundos: 45},
    visIA: {btn: 'ia-vision-btn', st: 'ia-vision-st', sk: 'ia-vision-sk',
          ver: ['ia-vision'], texto: 'mirando la foto…', segundos: 40},
    capIA: {btn: 'ia-cap-btn', st: 'ia-cap-st', sk: 'ia-cap-sk',
          ver: ['ia-cap'], texto: 'escribiendo el pie…', segundos: 90},
    comIA: {btn: 'ia-com-btn', st: 'ia-com-st', sk: 'ia-com-sk',
          ver: ['ia-coms'], texto: 'escribiendo…', segundos: 60},
  };
  function trabajando(clave, on) {
    const t = TRABAJOS[clave], b = $(t.btn), st = $(t.st);
    b.disabled = on;
    b.classList.toggle('trabajando', on);
    $(t.sk).style.display = on ? '' : 'none';
    t.ver.forEach(id => { if ($(id)) $(id).style.display = on ? 'none' : ''; });
    clearInterval(b._tick);   // el reloj vive en la línea de estado, no en el botón
    b._tick = null;
    if (!on) { st.textContent = ''; return; }
    // el timer va a la línea de estado (el botón queda con su ícono, dimmed):
    // así el botón puede ser un ✨ compacto sin que el reloj le rompa el ancho
    let s = 0;
    const pintar = () => { st.textContent = ` ${RELOJ[s % RELOJ.length]} ${t.texto} ${s}s`; s++; };
    pintar();
    b._tick = setInterval(pintar, 1000);
  }

  // --- pie de foto ---
  function genCaption() {
    const r = rel();
    if (!window.VLM_MODELS.length) {
      $('lb-cap-st').textContent = ' Ollama no está corriendo.'; return;
    }
    trabajando('cap', true);
    post_(CAP_URL, {rel: r, model: $('lb-model').value}).then(d => {
      trabajando('cap', false);
      $('lb-cap-st').textContent = d.ok ? '' : ' ' + d.error;
      if (!d.ok) return;
      slot(r).caption = d.caption;
      if (r === rel()) render();     // ya pasaste de foto: guardalo igual
    }).catch(e => { trabajando('cap', false); $('lb-cap-st').textContent = ' ' + e; });
  }

  // --- Centro de IA (modal "✨ Crear contenido") ---
  // Motor genérico: dispara la IA, marca el paso del acordeón como trabajando y
  // guarda SIEMPRE en la foto. render() repinta las dos vistas (columna + modal).
  function genIA(clave, url, guardar) {
    const r = rel();
    if (!window.VLM_MODELS.length) {
      $(TRABAJOS[clave].st).textContent = ' Ollama no está corriendo.'; return;
    }
    trabajando(clave, true);
    post_(url, {rel: r, model: $('lb-model').value}).then(d => {
      trabajando(clave, false);
      $(TRABAJOS[clave].st).textContent = d.ok ? ' ✓' : ' ' + d.error;
      if (!d.ok) return;
      const p = slot(r);
      guardar(p, d, r);
      if (r === rel()) render();     // ya pasaste de foto: guardalo igual
    }).catch(e => { trabajando(clave, false); $(TRABAJOS[clave].st).textContent = ' ' + e; });
  }
  function genVisionIA()  { genIA('visIA', VISION_URL,  (p, d) => { p.vision = d.vision; }); }
  function genCaptionIA() { genIA('capIA', CAP_URL,     (p, d) => { p.caption = d.caption; }); }
  function genCommentsIA(){ genIA('comIA', COM_URL,     (p, d) => { p.comments = d.comments; }); }
  function genCopeteIA()  { genIA('copIA', COPETE_URL,  (p, d, r) => { p.copete = d.copete;
                                                                        window.COPETES[r] = d.copete; }); }

  // "usar en la portada": el destacado vive SOLO en el navegador (localStorage),
  // así lo compartís con nadie; la home lo lee de ahí.
  function usarCopeteHomeIA() {
    const r = rel();
    const texto = (slot(r).copete || copeteDe(r) || '').trim();
    if (!texto) { $('ia-cop-st').textContent = ' Generá un copete primero (📰).'; return; }
    try {
      localStorage.setItem('home_copete',
        JSON.stringify({name: PERSON, rel: r, texto: texto, ts: Date.now()}));
      $('ia-cop-st').textContent = ' ✓ elegido para la portada';
    } catch (e) { $('ia-cop-st').textContent = ' ' + e; }
  }
  function openIA()  { $('ia-who').textContent = PERSON; $('lb-ia-modal').style.display = 'flex'; }
  function closeIA() { $('lb-ia-modal').style.display = 'none'; }

  // --- armar un molde de pie con las etiquetas de ESTA foto ---
  function abrirMoldeFoto() {
    const r = rel();
    const tags = PHOTO_TAGS[r] || [];             // solo las propias de la foto
    if (!tags.length) {
      alert('Esta foto todavía no tiene etiquetas propias. Etiquetala primero (🏷).'); return;
    }
    const nsfw = ('/' + r).split('/').includes('_nsfw');   // el registro sale de la carpeta
    $('lb-molde-modal').style.display = 'flex';
    $('lb-molde-json').value = ''; $('lb-molde-st').textContent = ' cargando…';
    // las del perfil (relación: hermana, mujer…) van aparte: sirven para
    // disparar el molde, pero no se le agregan a la foto
    post_(COMBO_URL, {tags: tags.join(','), perfil: (PERSON_TAGS || []).join(','),
                      nsfw: nsfw ? '1' : '0'}).then(d => {
      $('lb-molde-json').value = d.json || '';
      $('lb-molde-st').textContent = '';
    }).catch(e => $('lb-molde-st').textContent = ' ' + e);
  }
  function cerrarMoldeFoto() { $('lb-molde-modal').style.display = 'none'; }
  function copiarMoldeFoto() {
    const t = $('lb-molde-json'); t.select();
    navigator.clipboard.writeText(t.value).then(
      () => $('lb-molde-st').textContent = ' ✓ copiado',
      () => { document.execCommand('copy'); $('lb-molde-st').textContent = ' ✓ copiado'; });
  }
  function copiarPromptFoto() {
    if (!PROMPT_TPL || !PROMPT_TPL.trim()) {
      $('lb-molde-st').textContent = ' ✗ configurá el prompt en Ajustes primero'; return;
    }
    // "pelo largo", "buzo negro" — comillas y comas; escapo comillas internas
    const comillas = a => (a || []).map(t => '"' + String(t).replace(/"/g, '\\"') + '"').join(', ');
    const texto = PROMPT_TPL
      .split('$etiquetas').join(comillas(PHOTO_TAGS[rel()]))   // las de la foto
      .split('$perfil').join(comillas(PERSON_TAGS))            // las del perfil
      .split('$vinculo').join(VINCULO || '')                   // lo que ES ella: prima
      .split('$relacion').join(RELACION || '');                // cómo te llama: primo
    navigator.clipboard.writeText(texto).then(
      () => $('lb-molde-st').textContent = ' ✓ prompt copiado',
      () => { const ta = document.createElement('textarea'); ta.value = texto;
        document.body.appendChild(ta); ta.select(); document.execCommand('copy');
        ta.remove(); $('lb-molde-st').textContent = ' ✓ prompt copiado'; });
  }
  function guardarMoldeFoto() {
    const val = $('lb-molde-json').value;
    if (!val.trim()) { $('lb-molde-st').textContent = ' escribí algo primero'; return; }
    // no valido acá: el servidor es tolerante (saca cercas, comillas curvas,
    // comas colgando) y acepta emojis tal cual. Guarda el molde Y suma las
    // etiquetas del JSON a la foto (unión, nunca pisa).
    $('lb-molde-st').textContent = ' guardando…';
    post_(DESDE_FOTO_URL, {rel: rel(), texto: val}).then(d => {
      if (!d.ok) { $('lb-molde-st').textContent = ' ✗ ' + d.error; return; }
      if (Array.isArray(d.tags)) PHOTO_TAGS[rel()] = d.tags;   // reflejar las nuevas
      cerrarMoldeFoto();
      genCaption();     // regenera el pie de esta foto: ahora lo trae del molde
    }).catch(e => $('lb-molde-st').textContent = ' ' + e);
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
    const r = rel();
    if (!window.VLM_MODELS.length) {
      $('lb-com-st').textContent = ' Ollama no está corriendo.'; return;
    }
    trabajando('com', true);
    // si ya tiene comentarios, los nuevos se SUMAN a los que están
    post_(COM_URL, {rel: r, model: $('lb-model').value,
                    sumar: slot(r).comments.length ? '1' : ''}).then(d => {
      trabajando('com', false);
      $('lb-com-st').textContent = d.ok ? '' : ' ' + d.error;
      if (!d.ok) return;
      slot(r).comments = d.comments;
      if (r === rel()) render();
    }).catch(e => { trabajando('com', false); $('lb-com-st').textContent = ' ' + e; });
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
    box.querySelectorAll('.chip, .ownmore').forEach(c => c.remove());
    const own = ownTags();
    // muchas etiquetas tapan el resto del panel: se muestran las primeras
    // OWN_LIMIT y un "mostrar todas" para desplegar el resto
    const visibles = ownExpanded ? own : own.slice(0, OWN_LIMIT);
    visibles.forEach(t => {
      const c = document.createElement('span');
      c.className = 'chip';
      c.innerHTML = esc(t) + ' <span class="x">×</span>';
      c.querySelector('.x').onclick = () => saveOwn(ownTags().filter(x => x !== t));
      box.insertBefore(c, input);
    });
    if (own.length > OWN_LIMIT) {
      const b = document.createElement('button');
      b.className = 'ownmore';
      b.textContent = ownExpanded ? '▲ mostrar menos'
                                  : `▼ mostrar todas (${own.length})`;
      b.onclick = () => { ownExpanded = !ownExpanded; renderTags(); };
      box.insertBefore(b, input);
    }
    renderTagPreview();     // primeras 3 etiquetas + "ver más" en el panel lateral
    renderFavs();           // tus favoritas, a un clic
  }
  // Tus favoritas (de esta galería + globales) que la foto todavía NO tiene:
  // un clic y se agregan, para no tipear siempre lo mismo.
  function renderFavs() {
    const box = $('lb-favs'); if (!box) return;
    box.innerHTML = '';
    const own = ownTags();
    const libres = (FAVS || []).filter(t => !own.includes(t) && !PERSON_TAGS.includes(t));
    if (!libres.length) {
      box.innerHTML = '<span class="meta empty">' +
        ((FAVS || []).length ? 'ya están todas puestas'
                             : 'marcalas con ⭐ en ⚙ → Gestor de etiquetas') + '</span>';
      return;
    }
    libres.forEach(t => {
      const c = document.createElement('span'); c.className = 'chip';
      c.style.cursor = 'pointer'; c.textContent = t; c.title = 'Agregar a esta foto';
      c.onclick = () => saveOwn(ownTags().concat([t]));
      box.appendChild(c);
    });
  }
  // Panel lateral: muestra las primeras 3 etiquetas (propias + heredadas) como
  // chips, y un "ver más (N)" que despliega el resto inline.
  let _tagPrevExp = false;                 // "ver más" del preview (se resetea por foto)
  // Preview editable: chips PROPIAS con × (borrar) y clic para editar, un input
  // para agregar rápido, y las HEREDADAS de la persona apagadas (solo lectura).
  function pintarTagPrev(box) {
    if (!box) return;
    box.innerHTML = '';
    const own = ownTags();
    (_tagPrevExp ? own : own.slice(0, 3)).forEach(t => {
      const c = document.createElement('span'); c.className = 'chip';
      const txt = document.createElement('span'); txt.className = 'chiptxt'; txt.textContent = t;
      txt.title = 'clic para editar';
      txt.onclick = () => cpEditar(t, c);
      const x = document.createElement('span'); x.className = 'x'; x.textContent = '×';
      x.title = 'Quitar'; x.onclick = () => saveOwn(ownTags().filter(v => v !== t));
      c.appendChild(txt); c.appendChild(document.createTextNode(' ')); c.appendChild(x);
      box.appendChild(c);
    });
    if (own.length > 3) {
      const b = document.createElement('button'); b.className = 'linklike'; b.style.marginLeft = '.3rem';
      b.textContent = _tagPrevExp ? 'ver menos' : ('ver más (' + (own.length - 3) + ')');
      b.onclick = () => { _tagPrevExp = !_tagPrevExp; renderTagPreview(); };
      box.appendChild(b);
    }
    const inp = document.createElement('input');
    inp.className = 'tagmini'; inp.placeholder = '+ etiqueta'; inp.spellcheck = false;
    inp.onkeydown = (e) => {
      if (e.key === 'Enter') { e.preventDefault(); const v = inp.value.trim();
        if (v && !ownTags().includes(v)) saveOwn(ownTags().concat([v])); inp.value = ''; }
    };
    box.appendChild(inp);
    PERSON_TAGS.forEach(t => {
      const c = document.createElement('span'); c.className = 'chip inherited';
      c.textContent = t; c.title = 'heredada de la persona (se edita en el perfil)';
      box.appendChild(c);
    });
  }
  function renderTagPreview() {
    pintarTagPrev($('lb-tag-preview'));      // panel lateral del post
    pintarTagPrev($('ia-tag-preview'));      // paso 2 del Centro de IA
  }
  function cpEditar(viejo, chip) {         // editar una etiqueta propia en el lugar
    const inp = document.createElement('input');
    inp.className = 'tagmini'; inp.value = viejo; inp.spellcheck = false;
    chip.replaceWith(inp); inp.focus(); inp.select();
    const commit = () => {
      const v = inp.value.trim();
      if (v && v !== viejo) saveOwn(ownTags().map(x => x === viejo ? v : x));
      else renderTagPreview();
    };
    inp.onkeydown = (e) => {
      if (e.key === 'Enter') { e.preventDefault(); commit(); }
      else if (e.key === 'Escape') renderTagPreview();
    };
    inp.onblur = commit;
  }
  // Enter/Escape los maneja tagAC entero (ver BASE_JS): un solo lugar, así no
  // se duplica la etiqueta cuando hay una sugerencia resaltada.
  tagAC($('lb-tin'), {commit: v => { if (!ownTags().includes(v)) saveOwn(ownTags().concat([v])); }});
  function openTagModal() {
    renderTags();                          // refresca aplicadas
    $('tagm-photo').src = P[idx].url;      // la foto actual, para chequear contra ella
    $('lb-tag-modal').style.display = 'flex';
  }
  function closeTagModal() { $('lb-tag-modal').style.display = 'none'; }
  // --- sugerir etiquetas con IA ---
  // Hay DOS botones que disparan lo mismo: el del gestor/panel lateral (🔮 `lb-sug`)
  // y el del paso 2 del Centro de IA (🔮 `ia-sug`). Cada uno vive en su propio
  // contenedor para que no se pisen las sugerencias; comparten toda la lógica.
  function crearSugestador(b) {
    // b = {btn, st, sk, tags, count}: ids de los elementos de su panel
    const btn = () => $(b.btn), st = () => $(b.st), sk = () => $(b.sk),
          tags = () => $(b.tags), count = () => $(b.count);
    let sugeridas = new Set();   // ya sugeridas por IA ("no repetir" ni descartadas)
    let sugStop = false;         // el usuario apretó Parar o cambió de foto
    let sugCorriendo = false;    // hay una corrida en curso en ESTE panel
    // agrega una sugerencia: un chip con "+ tag" (clic = agregar) y un ✕
    // (clic = descartar). Descartada o agregada, queda en `sugeridas`, así el
    // próximo "Sugerir" NO la vuelve a traer. nueva=true la marca en ámbar.
    function addSugBtn(tag, nueva, acto) {
      // si tenés una favorita que dice lo mismo, se usa TU palabra (no la de la IA)
      const dijoIA = tag;
      const mia = favParecida(tag);
      if (mia) { tag = mia; nueva = false; }
      const chip = document.createElement('span');
      chip.className = 'sugchip' + (nueva ? ' sug-nueva' : '') + (acto ? ' sug-acto' : '');
      // marcador ⭐/🎯/⚠/+ · input EDITABLE (podés retocarla antes de agregarla)
      const mark = document.createElement('span');
      mark.className = 'sugmark';
      mark.textContent = mia ? '⭐' : (acto ? '🎯' : (nueva ? '⚠' : '+'));
      mark.title = mia ? ('Tu favorita (la IA había dicho «' + dijoIA + '»)')
                : (acto ? 'Detectado por el clasificador de actos (más confiable que el VLM)'
                : (nueva ? 'Palabra rara: fijate que no sea invento del modelo' : 'Sugerencia'));
      const inp = document.createElement('input');
      inp.className = 'suginput'; inp.value = tag; inp.spellcheck = false;
      const agregar = () => {
        const v = inp.value.trim();
        if (v && !ownTags().includes(v)) saveOwn(ownTags().concat([v]));
        chip.remove(); actualizarSugCount();
      };
      inp.onkeydown = (e) => { if (e.key === 'Enter') { e.preventDefault(); agregar(); } };
      const add = document.createElement('button');
      add.className = 'sugadd2'; add.textContent = '＋'; add.title = 'Agregar';
      add.onclick = agregar;
      const x = document.createElement('button');
      x.className = 'sugx'; x.textContent = '✕';
      x.title = 'Descartar (no volver a sugerirla)';
      x.onclick = () => { chip.remove(); actualizarSugCount(); };  // queda en sugeridas
      chip.appendChild(mark); chip.appendChild(inp); chip.appendChild(add); chip.appendChild(x);
      tags().appendChild(chip);
    }
    function actualizarSugCount() {
      const n = tags().querySelectorAll('.sugchip').length;
      count().textContent = n ? `${n} sugerencia(s) · + agregar · ✕ descartar` : '';
    }
    // UNA pasada de streaming: pide etiquetas (excluyendo lo ya visto), las
    // APPENDEA a la lista en vivo, y devuelve cuántas NUEVAS trajo. El server
    // sube la temp y le pide "escarbar" cuando `evitar` no viene vacío.
    async function unaPasada() {
      const r = rel();
      // se le pide que NO repita lo que ya tiene la foto NI lo que ya es general
      // del perfil (si "ojos azules" es general, no tiene sentido sugerirla de nuevo)
      const evitar = [...new Set([...sugeridas, ...ownTags(), ...PERSON_TAGS])];
      let nuevas = 0;
      try {
        // mando el modelo elegido en el gestor (el server cae a tag_model() si no)
        const modelo = ($('lb-tagmodel') || {}).value || '';
        const resp = await fetch(SUG_STREAM_URL, {method: 'POST',
          headers: {'Content-Type': 'application/x-www-form-urlencoded'},
          body: new URLSearchParams({rel: r, evitar: evitar.join(','), vlm_model: modelo})});
        const reader = resp.body.getReader();
        const dec = new TextDecoder();
        let buf = '';
        while (true) {
          const {done, value} = await reader.read();
          if (done) break;
          if (r !== rel() || sugStop) { reader.cancel(); break; }
          buf += dec.decode(value, {stream: true});
          let nl;
          while ((nl = buf.indexOf('\\n')) >= 0) {
            const linea = buf.slice(0, nl); buf = buf.slice(nl + 1);
            if (!linea.trim()) continue;
            let d; try { d = JSON.parse(linea); } catch (e) { continue; }
            if (d.error) { st().textContent = ' ' + d.error; continue; }
            const tag = d.tag;
            // el acto (🎯) nunca se filtra por parecido: es anclado y va sí o sí
            if (!tag || sugeridas.has(tag) || ownTags().includes(tag)
                || PERSON_TAGS.includes(tag)) continue;
            if (!d.acto && parecida(tag, sugeridas)) { sugeridas.add(tag); continue; }  // parecida: la recuerdo pero no la muestro
            sk().style.display = 'none';
            sugeridas.add(tag);
            nuevas++;
            addSugBtn(tag, d.nueva, d.acto);
            actualizarSugCount();
          }
        }
      } catch (e) {
        st().textContent = ' ' + e;
      }
      return nuevas;
    }
    // AUTO-CONTINUO: apretás una vez y sigue pidiendo tanda tras tanda EN VIVO,
    // acumulando, para que leas lo que va cayendo sin apretar "más". Para cuando:
    // apretás Parar (segundo click), 2 rondas seguidas sin nada nuevo, o el tope.
    // El mismo botón hace de "Sugerir" y de "Parar".
    const SUG_MAX_RONDAS = 5;   // pocas: más allá, el modelo empieza a inventar
    async function correr() {
      if (!window.VLM_MODELS.length) {
        st().textContent = ' Ollama no está corriendo.'; return;
      }
      if (sugCorriendo) { sugStop = true; return; }   // segundo click = Parar
      const startRel = rel();
      sugCorriendo = true; sugStop = false;
      sugeridas = new Set(); tags().innerHTML = ''; actualizarSugCount();  // fresco
      sk().style.display = '';
      btn().textContent = '⏹ Parar';
      btn().classList.add('trabajando');
      let dry = 0, rondas = 0;
      while (!sugStop && rel() === startRel && rondas < SUG_MAX_RONDAS && dry < 2) {
        const n = await unaPasada();
        rondas++;
        dry = n === 0 ? dry + 1 : 0;
        st().textContent = ` ${sugeridas.size} etiquetas · buscando más…`;
      }
      sk().style.display = 'none';
      sugCorriendo = false;
      if (rel() === startRel) {                       // no cambiaste de foto
        btn().textContent = '🔮 Sugerir con IA';
        btn().classList.remove('trabajando');
        st().textContent = sugeridas.size ? ` ${sugeridas.size} etiquetas`
                                          : ' no encontró nada';
      }
    }
    function reset() {          // cambió la foto: cortá cualquier corrida en curso
      sugeridas = new Set();
      sugStop = true;
      if (btn()) { btn().textContent = '🔮 Sugerir con IA'; btn().classList.remove('trabajando'); }
      if (sk()) sk().style.display = 'none';
      if (st()) st().textContent = '';
    }
    return {correr, reset};
  }
  const sugestLB = crearSugestador({btn: 'lb-sug', st: 'lb-sug-st', sk: 'lb-sug-sk',
                                    tags: 'lb-sugtags', count: 'lb-sug-count'});
  const sugestIA = crearSugestador({btn: 'ia-sug', st: 'ia-sug-st', sk: 'ia-sug-sk',
                                    tags: 'ia-sugtags', count: 'ia-sug-count'});

  // ¿la etiqueta es PARECIDA a alguna ya vista/agregada/descartada? Compara por
  // palabras significativas (sin "de/en/la…", plural≈singular): si las de una son
  // subconjunto de las de la otra, es la misma idea. Así "aretes plateado" tapa a
  // "aretes plateados en las dos orejas": si descarté la primera, no quiero la otra.
  const _STOP = new Set(['de','en','la','el','las','los','un','una','unos','unas',
    'con','y','a','al','del','su','sus','se','lo','le','dos','tres','muy','esta','este']);
  function _palabras(t) {
    return t.toLowerCase().normalize('NFD').replace(/[\\u0300-\\u036f]/g, '')
      .replace(/[^a-z0-9\\s]/g, ' ')          // "pelo:" cuenta como "pelo"
      .split(/\\s+/).map(w => w.replace(/s$/, '')).filter(w => w && !_STOP.has(w));
  }
  // ¿alguna favorita TUYA dice lo mismo que esta sugerencia? Ej: la IA dice
  // "tipo de pelo: lacio" y vos tenés "pelo lacio" -> gana la tuya.
  function favParecida(tag) {
    const a = new Set(_palabras(tag));
    if (!a.size) return null;
    for (const f of (FAVS || [])) {
      if (f.toLowerCase() === tag.toLowerCase()) return null;   // ya es idéntica
      const b = new Set(_palabras(f));
      if (!b.size) continue;
      const [chico, grande] = b.size <= a.size ? [b, a] : [a, b];
      let dentro = true;
      for (const w of chico) if (!grande.has(w)) { dentro = false; break; }
      if (dentro) return f;
    }
    return null;
  }
  function parecida(tag, vistas) {   // `vistas` = las ya sugeridas de ESTE panel
    const a = new Set(_palabras(tag));
    if (!a.size) return false;
    for (const s of [...vistas, ...ownTags(), ...PERSON_TAGS]) {
      const b = new Set(_palabras(s));
      if (!b.size) continue;
      const [chico, grande] = a.size <= b.size ? [a, b] : [b, a];
      let dentro = true;
      for (const w of chico) if (!grande.has(w)) { dentro = false; break; }
      if (dentro) return true;   // una es subconjunto de la otra -> misma idea
    }
    return false;
  }

  Object.assign(window, {genCaption, editCap, genComments, clearComments,
                         openTagModal, closeTagModal,
                         suggestTags: sugestLB.correr, sugIA: sugestIA.correr,
                         abrirMoldeFoto, cerrarMoldeFoto, copiarMoldeFoto, guardarMoldeFoto,
                         copiarPromptFoto,
                         genVisionIA, genCaptionIA, genCommentsIA, genCopeteIA,
                         usarCopeteHomeIA, openIA, closeIA});

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
    if (e.key === 'Escape' && $('lb-people-modal').style.display !== 'none') {
      closePeopleModal(); return;
    }
    if (e.key === 'Escape' && $('lb-tag-modal').style.display !== 'none') {
      closeTagModal(); return;
    }
    if (e.key === 'Escape' && $('lb-molde-modal').style.display !== 'none') {
      cerrarMoldeFoto(); return;
    }
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
    # Videos: SOLO para ver/reproducir en la galería. No entran al pipeline de
    # etiquetas/pie/comentarios (eso es sobre imágenes), por eso van aparte y no
    # se suman a normal/nsfw_list. Siguen el mismo toggle NSFW.
    vids = person_photos(known_dir, name, media="videos") if known_dir else []
    vids_nsfw = person_photos(known_dir, name, only_nsfw=True, media="videos") if known_dir else []
    show_nsfw = request.args.get("nsfw") == "1"
    # El toggle es EXCLUYENTE: normal = solo las normales; "Solo NSFW" = solo las
    # NSFW (no las mezcla). Cada foto lleva sus etiquetas efectivas para filtrar.
    todos_posts = posts_mod.get_many(name, normal + nsfw_list)
    def _fav(r):
        return todos_posts.get(r, {}).get("fav", False)
    if show_nsfw:
        display = [{"rel": r, "nsfw": True, "video": False,
                    "tags": tags_mod.effective_photo(name, r), "fav": _fav(r)} for r in nsfw_list]
        display += [{"rel": r, "nsfw": True, "video": True, "tags": [], "fav": False}
                    for r in vids_nsfw]
    else:
        display = [{"rel": r, "nsfw": False, "video": False,
                    "tags": tags_mod.effective_photo(name, r), "fav": _fav(r)} for r in normal]
        display += [{"rel": r, "nsfw": False, "video": True, "tags": [], "fav": False}
                    for r in vids]
    photos = normal   # para los avisos "no hay imágenes / configurá carpeta"

    # El filtro por etiquetas es del SERVIDOR, no del navegador: si filtrara la
    # página ya paginada te mostraría "3 fotos" cuando en realidad hay 40 en el
    # resto de las páginas.
    filter_tags = request.args.get("tags", "")
    buscadas = [t for t in (_norm_search(t) for t in filter_tags.split(",")) if t]
    if buscadas:
        display = [f for f in display
                   if all(any(t in _norm_search(x) for x in f["tags"]) for t in buscadas)]
    # filtro por co-aparición: solo las fotos donde TAMBIÉN está esta persona
    filter_con = request.args.get("con", "")
    if filter_con:
        con_posts = posts_mod.get_many(name, [f["rel"] for f in display])
        display = [f for f in display
                   if filter_con in con_posts[f["rel"]]["people"]]
    # filtro por favoritos (⭐)
    solo_fav = request.args.get("fav") == "1"
    if solo_fav:
        display = [f for f in display if f["fav"]]
    # filtro por ESTADO: para ir tachando pendientes en vez de buscarlos a ojo
    estado = request.args.get("estado", "")
    if estado:
        hechas_ia = procesadas.de_persona(name)
        posts_est = posts_mod.get_many(name, [f["rel"] for f in display])
        def pasa(f):
            r = f["rel"]
            if estado == "sin_etiquetas":
                return not tags_mod.get_photo(name, r)
            if estado == "sin_ia":
                return r not in hechas_ia
            if estado == "ia_vacia":            # pasó por la IA y no trajo nada
                return hechas_ia.get(r, {}).get("n") == 0
            if estado == "sin_pie":
                return not posts_est[r]["caption"]
            return True
        display = [f for f in display if pasa(f)]
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
    # el vínculo de esta persona (hermana, prima…), del perfil
    _vinc_persona = moldes_mod.relacion_de(tags_mod.get_person(name)) or ""
    return render_template_string(
        shell("known", PERSON_TEMPLATE),
        name=name,
        # el pie y los comentarios ya guardados de cada foto: el modal los
        # muestra al instante, sin esperar al modelo
        posts=posts_mod.get_many(name, [f["rel"] for f in display]),
        # copetes (titulares de noticia) ya generados por foto: {rel: texto}
        copetes=dict(copetes_mod.por_persona(name)),
        # favoritos de la persona (para el carrusel de la card)
        favs=[r for r in normal + nsfw_list if todos_posts.get(r, {}).get("fav")],
        # para el lote: cuántas le faltan de TODAS las suyas (no de esta página)
        n_sin_post=sum(1 for p in posts_mod.get_many(
            name, normal + nsfw_list).values() if not p["caption"]),
        display_total=len(normal) + len(nsfw_list),
        lote=POSTS_STATE,
        # si Ollama no está corriendo, la lista viene vacía y el modal lo avisa
        vlm_models=vlm.list_models(),
        tag_modelo=vlm.tag_model(),        # el que usa el sugeridor por defecto
        vlm_crudos=vlm.modelos_descensurados(),  # los sin censura (se marcan 🔞)
        people=people,       # para el desplegable "mover a otra persona"
        person_tags=tags_mod.get_person(name),
        # tus favoritas para esta persona (las suyas + las globales): para
        # etiquetar rápido a mano y para pisar la redacción de la IA
        favs_tags=tags_mod.favs_efectivas(name),
        photo_tags=photo_tags,
        prompt_externo=load_config().get("prompt_externo", ""),
        # variables de vínculo para el prompt, con el MISMO sentido que en los
        # moldes: $vinculo = lo que ES ella; $relacion = cómo te llama a vos
        vinculo=_vinc_persona,
        relacion=(moldes_mod.relacion_para_duenio(_vinc_persona) or "") if _vinc_persona else "",
        all_tags=tags_mod.all_tags(),
        filter_tags=filter_tags,       # pre-filtro que viene de la galería
        filter_con=filter_con,         # filtro por co-aparición (👥 también aparece X)
        solo_fav=solo_fav,             # filtro por favoritos (⭐)
        estado=estado,                 # filtro por lo que falta hacer
        vinculo_actual=_vinc_persona,  # el vínculo que tiene puesto hoy
        vinculos_posibles=moldes_mod.vinculos_conocidos(),
        photos=photos,
        display=display,
        # paginado
        page=page, n_pages=n_pages, per=per, per_options=PER_OPTIONS,
        n_filtradas=n_filtradas,
        n_todas=(len(nsfw_list) + len(vids_nsfw)) if show_nsfw else (len(normal) + len(vids)),
        desde=(page - 1) * per + 1 if display else 0,
        hasta=(page - 1) * per + len(display),
        nsfw_count=len(nsfw_list),
        show_nsfw=show_nsfw,
        folder=str(Path(known_dir) / name) if known_dir else "",
        # de qué foto salió la portada (para que el editor abra ahí directo)
        cover_src=person_meta.get_cover_src(name),
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
        bio=meta["bio"],
        pick_rel=pick_rel,
        pick_count=pick_count,
        pick_v=int(time.time()),
        message=request.args.get("msg", ""),
    )


def _sniff_mime(path):
    """Tipo de imagen por su CONTENIDO (magic bytes), no por la extensión. Un
    archivo con extensión mentida (un JPEG llamado .webp) servido con el tipo
    equivocado se ve roto en el navegador; con el tipo real, anda. None si no lo
    reconoce (ahí se cae a la extensión, como antes)."""
    try:
        with open(path, "rb") as f:
            head = f.read(16)
    except OSError:
        return None
    if head[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if head[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if head[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image/webp"
    if head[:2] == b"BM":
        return "image/bmp"
    if head[4:8] == b"ftyp":                      # contenedor ISO-BMFF: avif/heic
        brand = head[8:12]
        if brand == b"avif":
            return "image/avif"
        if brand in (b"heic", b"heix", b"mif1", b"msf1", b"heif"):
            return "image/heic"
    return None


@app.route("/person_photo/<name>/<path:rel>")
def person_photo(name, rel):
    """Sirve la foto ORIGINAL (para el visor con zoom). El tipo MIME se detecta
    por el contenido: así una foto con extensión mentida (ej: un JPEG guardado
    como .webp) igual se muestra bien en el visor en vez de romperse."""
    cfg = load_config()
    directory = Path(get_library(cfg)) / name
    mt = _sniff_mime(directory / rel)
    if mt:
        return send_from_directory(directory, rel, mimetype=mt)
    return send_from_directory(directory, rel)


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
            if src.suffix.lower() in VIDEO_EXTENSIONS:
                im = _video_frame_pil(src)            # miniatura = 1er fotograma
                if im is None:
                    abort(404)
            else:
                im = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
            im.thumbnail((360, 360))
            im.save(cache, quality=82)
        except Exception:
            return send_from_directory(Path(library) / name, rel)  # ilegible: el original
    return send_from_directory(GALLERY_THUMBS, key)


def _recorte_cuadrado(im, box):
    """Cuadrado (el lado más grande permite que quepa) alrededor de un box de
    cara, o del centro de la imagen si box es None. Imagen PIL RGB -> PIL."""
    w, h = im.size
    if box:
        top, right, bottom, left = box
        cx = (left + right) / 2
        cy = (top + bottom) / 2
        side = max(right - left, bottom - top) * 1.35
        side = min(side, w, h)
    else:
        cx, cy = w / 2, h / 2
        side = min(w, h)
    x0, y0 = cx - side / 2, cy - side / 2
    if x0 < 0:
        x0 = 0
    if y0 < 0:
        y0 = 0
    if x0 + side > w:
        x0 = w - side
    if y0 + side > h:
        y0 = h - side
    crop = im.crop((x0, y0, x0 + side, y0 + side))
    crop.thumbnail((360, 360))
    return crop


def _cara_thumb(im, name):
    """Miniatura 360 centrada en la cara de <name> dentro de <im> (PIL RGB).

    Una sola cara -> recorte alrededor de esa cara. Más de una cara (foto
    grupal) -> la foto de perfil de la persona (KNOWN_THUMBS/<name>.jpg) si
    existe (en la home manda el perfil), y si no, un cuadrado del centro de la
    foto. Sin caras -> None (el llamador cae a la miniatura clásica)."""
    im = im.copy()
    im.thumbnail((512, 512))
    from app import facedet
    import numpy as _np
    faces = facedet.detect_and_encode(_np.array(im))
    if len(faces) > 1:
        cover = KNOWN_THUMBS / f"{safe_name(name)}.jpg"
        if cover.is_file():
            with Image.open(cover) as c:
                cb = c.convert("RGB").copy()
                cb.thumbnail((360, 360))
                return cb
        return _recorte_cuadrado(im, None)
    if len(faces) == 1:
        return _recorte_cuadrado(im, faces[0][0])
    return None


@app.route("/person_hero_thumb/<name>/<path:rel>")
def person_hero_thumb(name, rel):
    """Miniatura de la HOME centrada en la cara de la persona.

    Detrás de /person_thumb (que recorta el centro de la foto) agrega el
    recorte por rostro: la foto de la home muestra a esta persona, no un
    costado de la imagen. Regla: con una sola cara se centra ahí; con más de
    una (foto grupal) se usa la foto de perfil de la persona; si no hay caras o
    algo falla, cae a la miniatura clásica. Cacheada una vez por foto."""
    library = get_library()
    src = Path(library) / name / rel if library else None
    if not src or not src.is_file():
        abort(404)
    key = "hero_" + hashlib.sha1(f"{name}/{rel}".encode("utf-8")).hexdigest() + ".jpg"
    cache = GALLERY_THUMBS / key
    if not cache.is_file() or cache.stat().st_mtime < src.stat().st_mtime:
        GALLERY_THUMBS.mkdir(parents=True, exist_ok=True)
        out = None
        try:
            if src.suffix.lower() in VIDEO_EXTENSIONS:
                im = _video_frame_pil(src)
                if im is not None:
                    out = _cara_thumb(im.convert("RGB"), name)
            else:
                im = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
                out = _cara_thumb(im, name)
        except Exception:
            out = None
        if out is None:
            try:
                im = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
                im.thumbnail((360, 360))
                out = im
            except Exception:
                return send_from_directory(Path(library) / name, rel)
        out.save(cache, quality=82)
    return send_from_directory(GALLERY_THUMBS, key)


def _wide_hero(im, aspect=2.6, face_y=0.34):
    """Banda ancha (aspect) de la foto para el banner del hero, con la cara de
    la persona a ~face_y del alto (por defecto 1/3, arriba del pie/título).
    Sin cara o con varias (foto grupal) usa una banda central: así el hero
    nunca queda dominado por un rostro gigante ni borroso de la portada.
    Devuelve PIL RGB."""
    im = im.copy()
    im.thumbnail((1280, 1280))
    from app import facedet
    import numpy as _np
    faces = facedet.detect_and_encode(_np.array(im))
    box = faces[0][0] if len(faces) == 1 else None
    w, h = im.size
    crop_w = min(w, int(h * aspect))
    crop_h = int(crop_w / aspect)
    if box:
        top, right, bottom, left = box
        cx = (left + right) / 2
        cy = (top + bottom) / 2
        y_top = cy - crop_h * face_y
    else:
        cx, cy = w / 2, h / 2
        y_top = cy - crop_h / 2
    x0 = max(0, int(min(w - crop_w, cx - crop_w / 2)))
    y0 = max(0, int(min(h - crop_h, y_top)))
    crop = im.crop((x0, y0, x0 + crop_w, y0 + crop_h))
    if crop.width < 1280:
        crop = crop.resize((1280, max(1, int(1280 / aspect))), Image.LANCZOS)
    return crop


@app.route("/person_hero_wide/<name>/<path:rel>")
def person_hero_wide(name, rel):
    """Imagen ANCHA del hero de la home: banda con el aspect del banner (2.6/1),
    con la cara de la persona cerca del tercio superior. Se usa SOLO en el
    carrusel del hero (donde un crop cuadrado de cara se ve horrible); las
    tarjetas siguen con person_hero_thumb. Cacheada por foto."""
    library = get_library()
    src = Path(library) / name / rel if library else None
    if not src or not src.is_file():
        abort(404)
    key = "herow_" + hashlib.sha1(f"{name}/{rel}".encode("utf-8")).hexdigest() + ".jpg"
    cache = GALLERY_THUMBS / key
    if not cache.is_file() or cache.stat().st_mtime < src.stat().st_mtime:
        GALLERY_THUMBS.mkdir(parents=True, exist_ok=True)
        out = None
        try:
            if src.suffix.lower() in VIDEO_EXTENSIONS:
                im = _video_frame_pil(src)
                if im is not None:
                    out = _wide_hero(im.convert("RGB"))
            else:
                im = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
                out = _wide_hero(im)
        except Exception:
            out = None
        if out is None:
            try:
                im = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
                im.thumbnail((360, 360))
                out = im
            except Exception:
                return send_from_directory(Path(library) / name, rel)
        out.save(cache, quality=82)
    return send_from_directory(GALLERY_THUMBS, key)


@app.route("/person_hero_fit/<name>/<path:rel>")
def person_hero_fit(name, rel):
    """Recorte de la home que "entra al ancho" (mosaico Agregadas recientemente):
    banda centrada en la cara al aspect pedido (a=1..2.6, default 1.35) con el
    rostro al centro del alto — se ve la foto entera a lo ancho, sin que la
    cara explote. Cacheada por foto y aspect."""
    a = 1.35
    try:
        a = float(request.args.get("a", 1.35))
        a = min(2.6, max(1.0, a))
    except (TypeError, ValueError):
        pass
    library = get_library()
    src = Path(library) / name / rel if library else None
    if not src or not src.is_file():
        abort(404)
    atag = str(round(a, 2))
    key = f"herof_{atag}_" + hashlib.sha1(f"{name}/{rel}".encode("utf-8")).hexdigest() + ".jpg"
    cache = GALLERY_THUMBS / key
    if not cache.is_file() or cache.stat().st_mtime < src.stat().st_mtime:
        GALLERY_THUMBS.mkdir(parents=True, exist_ok=True)
        out = None
        try:
            if src.suffix.lower() in VIDEO_EXTENSIONS:
                im = _video_frame_pil(src)
                if im is not None:
                    out = _wide_hero(im.convert("RGB"), aspect=a, face_y=0.5)
            else:
                im = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
                out = _wide_hero(im, aspect=a, face_y=0.5)
        except Exception:
            out = None
        if out is None:
            try:
                im = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
                im.thumbnail((360, 360))
                out = im
            except Exception:
                return send_from_directory(Path(library) / name, rel)
        out.save(cache, quality=82)
    return send_from_directory(GALLERY_THUMBS, key)


def _video_frame_pil(path):
    """Primer fotograma legible de un video como imagen PIL RGB, o None."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    try:
        ok, frame = cap.read()
        if not ok or frame is None:
            return None
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()


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


@app.route("/post/<name>/nsfw", methods=["POST"])
def post_mark_nsfw(name):
    """Marca UNA foto como NSFW: la mueve a <persona>/_nsfw/. Así corre el
    clasificador de acto (🎯) sin tener que pasar por el escáner de carpeta."""
    library = get_library()
    rel = request.form.get("rel", "")
    if not library or not rel:
        return {"ok": False, "error": "Falta la foto."}
    if NSFW_DIR_NAME in Path(rel).parts:
        return {"ok": False, "error": "Ya está marcada como NSFW."}
    moved, errors = move_to_nsfw(Path(library) / name, [rel])
    if not moved:
        return {"ok": False, "error": errors[0][1] if errors else "No se pudo mover."}
    # las etiquetas y el pie/comentarios acompañan a la foto a su nueva ruta
    new_rel = f"{NSFW_DIR_NAME}/{rel.replace('/', '_')}"
    try:
        tags_mod.move_photo_tags(name, rel, name, new_rel)
        posts_mod.move_photo(name, rel, new_rel)
    except Exception:
        pass
    _undo_push({"tipo": "nsfw", "desc": f"marcar NSFW «{Path(rel).name}»",
                "name": name, "new_rel": new_rel, "orig_rel": rel})
    # el front actualiza en el lugar (sin recargar): necesita la ruta y url nuevas
    return {"ok": True, "new_rel": new_rel,
            "new_url": url_for("person_photo", name=name, rel=new_rel)}


# --- Deshacer: pila global de acciones reversibles (en memoria, se pierde al
# reiniciar). Guarda las últimas; "↩ Deshacer" (botón flotante) revierte la última.
UNDO_STACK = []
UNDO_MAX = 30


def _undo_push(entry):
    UNDO_STACK.append(entry)
    del UNDO_STACK[:-UNDO_MAX]        # no dejar crecer sin límite


def _undo_nsfw(e):
    """Revierte un 'marcar NSFW': la foto vuelve de _nsfw a su ruta original,
    con sus etiquetas y su pie."""
    import shutil
    library = get_library()
    if not library:
        return False, "No hay Biblioteca."
    src = Path(library) / e["name"] / e["new_rel"]
    dest = Path(library) / e["name"] / e["orig_rel"]
    if not src.is_file():
        return False, "Esa foto ya no está en _nsfw."
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return False, "Ya hay una foto en la ruta original."
    shutil.move(str(src), str(dest))
    try:
        tags_mod.move_photo_tags(e["name"], e["new_rel"], e["name"], e["orig_rel"])
        posts_mod.move_photo(e["name"], e["new_rel"], e["orig_rel"])
    except Exception:
        pass
    return True, f"«{Path(e['orig_rel']).name}» volvió a la galería normal."


def _undo_tags(e):
    """Revierte un guardado de etiquetas: restaura las que tenía la foto antes."""
    try:
        tags_mod.set_photo(e["name"], e["rel"], e["prev"])
        return True, f"Etiquetas de «{Path(e['rel']).name}» restauradas."
    except Exception as ex:
        return False, str(ex)


def _undo_tag_masiva(e):
    """Revierte un borrado masivo del gestor: vuelve a poner la etiqueta en las
    fotos que la tenían."""
    try:
        n = tags_mod.add_photo_tag_many(e["name"], e["rels"], e["tag"])
        return True, f"«{e['tag']}» restaurada en {n} foto(s)."
    except Exception as ex:
        return False, str(ex)


def _undo_general(e):
    """Revierte un 'hacer general': la saca del perfil y la devuelve a sus fotos."""
    try:
        propias = [t for t in tags_mod.get_person(e["name"]) if t != e["tag"]]
        tags_mod.set_person(e["name"], propias)
        n = tags_mod.add_photo_tag_many(e["name"], e["rels"], e["tag"])
        return True, f"«{e['tag']}» ya no es general (volvió a {n} foto/s)."
    except Exception as ex:
        return False, str(ex)


_UNDO_FN = {"nsfw": _undo_nsfw, "tags": _undo_tags, "tag_masiva": _undo_tag_masiva,
            "general": _undo_general}


@app.route("/undo/estado")
def undo_estado():
    """¿Hay algo para deshacer? Devuelve la descripción de lo último."""
    if UNDO_STACK:
        return {"puede": True, "desc": UNDO_STACK[-1]["desc"]}
    return {"puede": False, "desc": ""}


@app.route("/undo", methods=["POST"])
def undo():
    """Deshace la última acción reversible (marcar NSFW o guardar etiquetas)."""
    if not UNDO_STACK:
        return {"ok": False, "error": "Nada para deshacer."}
    e = UNDO_STACK.pop()
    fn = _UNDO_FN.get(e["tipo"])
    if not fn:
        return {"ok": False, "error": "No sé deshacer eso."}
    ok, msg = fn(e)
    if not ok:
        UNDO_STACK.append(e)         # falló: lo dejo para reintentar
    return {"ok": ok, "msg": msg}


@app.route("/person/<name>/tags_data")
def person_tags_data(name):
    """Para el gestor de etiquetas: las del perfil (valen para todas sus fotos) y
    el recuento de las de foto (en cuántas aparece cada una)."""
    counts = tags_mod.person_photo_tag_counts(name)
    # más usadas primero; a igual cantidad, alfabético
    fotos = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return {"ok": True, "persona": tags_mod.get_person(name),
            "fotos": [{"tag": t, "n": n} for t, n in fotos],
            "fav_global": tags_mod.get_favs(None),
            "fav_persona": tags_mod.get_favs(name)}


@app.route("/person/<name>/tags_fav", methods=["POST"])
def person_tags_fav(name):
    """Marca/desmarca una etiqueta como favorita, global o de esta persona."""
    tag = request.form.get("tag", "").strip()
    ambito = request.form.get("ambito", "persona")
    if not tag:
        return {"ok": False, "error": "Falta la etiqueta."}
    tags_mod.toggle_fav(tag, None if ambito == "global" else name)
    return {"ok": True, "fav_global": tags_mod.get_favs(None),
            "fav_persona": tags_mod.get_favs(name)}


@app.route("/person/<name>/tags_promote", methods=["POST"])
def person_tags_promote(name):
    """Hace la etiqueta GENERAL (del perfil) y la saca de las fotos sueltas."""
    tag = request.form.get("tag", "").strip()
    if not tag:
        return {"ok": False, "error": "Falta la etiqueta."}
    rels = tags_mod.promote_to_person(name, tag)
    _undo_push({"tipo": "general", "desc": f"hacer general «{tag}»",
                "name": name, "tag": tag, "rels": rels})
    return {"ok": True, "quitadas": len(rels),
            "persona": tags_mod.get_person(name)}


@app.route("/person/<name>/tags_remove_all", methods=["POST"])
def person_tags_remove_all(name):
    """Saca una etiqueta de TODAS las fotos de la persona (limpieza después de
    etiquetar en lote con IA)."""
    tag = request.form.get("tag", "").strip()
    if not tag:
        return {"ok": False, "error": "Falta la etiqueta."}
    rels = tags_mod.remove_photo_tag(name, tag)
    if rels:      # se borra sin preguntar, pero queda deshacible
        _undo_push({"tipo": "tag_masiva",
                    "desc": f"borrar «{tag}» de {len(rels)} foto(s)",
                    "name": name, "tag": tag, "rels": rels})
    return {"ok": True, "quitadas": len(rels)}


@app.route("/person/<name>/tags_rename_all", methods=["POST"])
def person_tags_rename_all(name):
    """Renombra una etiqueta en TODAS las fotos de la persona (arreglo masivo:
    si la IA metió 'tipo de pelo: lacio' en 50 fotos, se corrige de una)."""
    old = request.form.get("old", "").strip()
    new = request.form.get("new", "").strip()
    if not old or not new:
        return {"ok": False, "error": "Falta la etiqueta."}
    if old.lower() == new.lower():
        return {"ok": False, "error": "Es la misma etiqueta."}
    n = tags_mod.rename_photo_tag(name, old, new)
    return {"ok": True, "cambiadas": n}


@app.route("/person/<name>/tags_person_set", methods=["POST"])
def person_tags_person_set(name):
    """Reemplaza las etiquetas DEL PERFIL (las que heredan todas sus fotos)."""
    try:
        tags = json.loads(request.form.get("tags", "[]"))
    except Exception:
        return {"ok": False, "error": "Etiquetas inválidas."}
    saved = tags_mod.set_person(name, [str(t) for t in tags])
    return {"ok": True, "tags": saved}


@app.route("/person/<name>/cover_crop", methods=["POST"])
def person_cover_crop(name):
    """Portada a medida: recorta la foto elegida al cuadro que marcó el usuario
    (left/top/side en píxeles reales) y la guarda como miniatura de la persona.
    No toca la base de caras (es solo la imagen de portada)."""
    library = get_library()
    rel = request.form.get("rel", "")
    if not library or not rel:
        return {"ok": False, "error": "Falta la foto."}
    src = Path(library) / name / rel
    if not src.is_file():
        return {"ok": False, "error": "Esa foto ya no está."}
    try:
        left = int(float(request.form.get("left", 0)))
        top = int(float(request.form.get("top", 0)))
        side = int(float(request.form.get("side", 0)))
    except ValueError:
        return {"ok": False, "error": "Recorte inválido."}
    try:
        im = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
        W, H = im.size
        side = max(16, min(side, W, H))
        left = max(0, min(left, W - side))
        top = max(0, min(top, H - side))
        crop = im.crop((left, top, left + side, top + side)).resize((220, 220), Image.LANCZOS)
        KNOWN_THUMBS.mkdir(parents=True, exist_ok=True)
        crop.save(KNOWN_THUMBS / f"{safe_name(name)}.jpg", quality=88)
    except Exception as e:
        return {"ok": False, "error": str(e)}
    # recordar de qué foto y con qué recorte salió: el editor abre ahí la próxima
    person_meta.set_cover_src(name, rel, left, top, side)
    return {"ok": True, "url": url_for("known_thumb", name=name)}


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


def _posts_worker(name, rels, que, cfg):
    """Escribe el pie y/o los comentarios de muchas fotos, de a una.

    De a UNA a propósito: Ollama corre en el mismo CPU, así que mandarle dos
    juntas no acelera nada, solo hace que tarden las dos.

    Saltea las que ya tienen lo pedido, así se puede cortar y retomar sin perder
    lo hecho ni pagarlo de nuevo."""
    library = get_library(cfg)
    modelo = cfg.get("post_model") or None
    crudeza = cfg.get("crudeness", posts_mod.CRUDENESS_DEFAULT)
    dim = bool(cfg.get("diminutivos"))
    for i, rel in enumerate(rels, 1):
        if POSTS_STATE["stop"]:
            break
        POSTS_STATE.update(current=i, photo=rel)
        path = Path(library) / name / rel
        if not path.is_file():
            POSTS_STATE["salteadas"] += 1
            continue
        ya = posts_mod.get(name, rel)
        falta_pie = que in ("pie", "ambos") and not ya["caption"]
        falta_com = que in ("comentarios", "ambos") and not ya["comments"]
        if not falta_pie and not falta_com:
            POSTS_STATE["salteadas"] += 1
            continue
        t0 = time.time()
        try:
            # el pie primero: si van los dos, los comentarios lo leen y le contestan
            if falta_pie:
                pie = posts_mod.generate_caption(name, rel, str(path), model=modelo,
                                                 crudeness=crudeza, diminutivos=dim)
                # los últimos pies, en vivo: para juzgar la calidad SIN esperar al
                # final ni recargar la galería (era la queja: "espero 2 hs para
                # saber si viene mal")
                POSTS_STATE["ultimos"] = ([f"{rel}: {pie}"] + POSTS_STATE["ultimos"])[:5]
            if falta_com:
                coms = posts_mod.generate_comments(
                    name, rel, str(path), model=modelo, crudeness=crudeza,
                    diminutivos=dim, tone=cfg.get("comment_tone", posts_mod.DEFAULT_TONE),
                    users=cfg.get("comment_users") or None)
                if not falta_pie and coms:      # solo comentarios: mostrar uno
                    POSTS_STATE["ultimos"] = ([f"{rel}: 💬 {coms[0]['text']}"]
                                              + POSTS_STATE["ultimos"])[:5]
            POSTS_STATE["hechas"] += 1
            # los tiempos reales de las últimas 20: con eso se estima lo que falta
            POSTS_STATE["segundos"] = (POSTS_STATE["segundos"] + [time.time() - t0])[-20:]
        except Exception as e:
            # una foto que falla no puede cortar una corrida de toda la noche
            POSTS_STATE["fallaron"] += 1
            POSTS_STATE["error"] = f"{rel}: {e}"
    POSTS_STATE["status"] = "stopped" if POSTS_STATE["stop"] else "done"


def _posts_eta():
    """Cuánto falta, en segundos, según lo que tardaron las últimas fotos.

    None si todavía no hay ninguna medida (los primeros segundos)."""
    s = POSTS_STATE
    if not s["segundos"]:
        return None
    promedio = sum(s["segundos"]) / len(s["segundos"])
    return int(promedio * max(0, s["total"] - s["current"]))


def _humano(segs):
    """"37 h 30 min", "2 min": el número crudo en segundos no le dice nada a nadie."""
    if segs is None:
        return "calculando…"
    if segs < 90:
        return f"{int(segs)} s"
    if segs < 5400:
        return f"{round(segs / 60)} min"
    h, m = divmod(int(segs / 60), 60)
    return f"{h} h {m} min" if m else f"{h} h"


@app.route("/post/<name>/lote", methods=["POST"])
def post_lote(name):
    """Arranca el lote: pies y/o comentarios de todas las fotos de la persona."""
    if POSTS_STATE["status"] == "running":
        return redirect(url_for("person_page", name=name, msg="Ya hay un lote corriendo."))
    library = get_library()
    if not library:
        return redirect(url_for("person_page", name=name))
    que = request.form.get("que", "pie")
    rels = person_photos(library, name) + person_photos(library, name, only_nsfw=True)
    if not rels:
        return redirect(url_for("person_page", name=name, msg="No tiene fotos."))
    POSTS_STATE.update(status="running", current=0, total=len(rels), photo="",
                       error="", name=name, que=que, hechas=0, salteadas=0,
                       fallaron=0, segundos=[], empezo=time.time(), stop=False,
                       ultimos=[])
    threading.Thread(target=_posts_worker, args=(name, rels, que, load_config()),
                     daemon=True).start()
    return redirect(url_for("person_page", name=name))


@app.route("/post/lote/stop", methods=["POST"])
def post_lote_stop():
    """Corta el lote. Lo hecho hasta acá queda guardado."""
    POSTS_STATE["stop"] = True
    return {"ok": True}


@app.route("/post/lote/estado")
def post_lote_estado():
    """Para que la página se actualice sola sin recargar."""
    s = POSTS_STATE
    return {"status": s["status"], "current": s["current"], "total": s["total"],
            "photo": s["photo"], "name": s["name"], "hechas": s["hechas"],
            "salteadas": s["salteadas"], "fallaron": s["fallaron"],
            "error": s["error"], "falta": _humano(_posts_eta()),
            "lleva": _humano(time.time() - s["empezo"] if s["empezo"] else 0),
            "ultimos": s["ultimos"]}


def _tags_worker(name, rels, cfg):
    """Le pone etiquetas a muchas fotos, de a una: mira cada foto con el modelo
    de visión y anota lo que ve (ver vlm.suggest_tags — sin vocabulario fijo).

    Saltea las que YA tienen etiquetas propias (no las heredadas de la persona,
    esas no cuentan), así se puede cortar y retomar sin repetir laburo. Suma a
    lo que hubiera, nunca reemplaza."""
    library = get_library(cfg)
    # el lote de etiquetado usa el mismo modelo "que mira bien" que el sugeridor
    modelo = vlm.tag_model()
    # salvaguarda: sin etiqueta de adulto el sugeridor NO describe cuerpo ni nada
    # sexual (igual que el pie y los comentarios). La duda cierra, no abre.
    # Va DENTRO del bucle: "adulta" puede estar en la foto y no en el perfil.
    for i, rel in enumerate(rels, 1):
        if TAGS_STATE["stop"]:
            break
        TAGS_STATE.update(current=i, photo=rel)
        path = Path(library) / name / rel
        if not path.is_file():
            TAGS_STATE["salteadas"] += 1
            continue
        propias = tags_mod.get_photo(name, rel)
        if propias:
            TAGS_STATE["salteadas"] += 1
            continue
        try:
            sano = posts_mod.identidad_foto(name, rel)["sano"]
            nuevas = vlm.suggest_tags(str(path), model=modelo,
                                      quien=name, crudo=not sano)
            if nuevas:
                tags_mod.set_photo(name, rel, propias + nuevas)
                TAGS_STATE["ultimos"] = ([f"{rel}: {', '.join(nuevas)}"] + TAGS_STATE["ultimos"])[:5]
            procesadas.marcar(name, rel, len(nuevas))   # aunque hayan sido 0
            TAGS_STATE["hechas"] += 1
        except Exception as e:
            # una foto que falla no puede cortar el lote entero
            TAGS_STATE["fallaron"] += 1
            TAGS_STATE["error"] = f"{rel}: {e}"
    TAGS_STATE["status"] = "stopped" if TAGS_STATE["stop"] else "done"


@app.route("/tags/<name>/lote", methods=["POST"])
def tags_lote(name):
    """Arranca el etiquetador automático: las fotos seleccionadas, o si no
    seleccionaste ninguna, TODAS las que no tienen etiquetas propias."""
    if TAGS_STATE["status"] == "running":
        return {"ok": False, "error": "Ya hay un etiquetado corriendo."}
    library = get_library()
    if not library:
        return {"ok": False, "error": "Falta configurar la Biblioteca."}
    elegidas = [r for r in request.form.get("rels", "").split("\n") if r.strip()]
    if elegidas:
        rels = elegidas
    else:
        todas = person_photos(library, name) + person_photos(library, name, only_nsfw=True)
        rels = [r for r in todas if not tags_mod.get_photo(name, r)]
    if not rels:
        return {"ok": False, "error": "No hay fotos para etiquetar."}
    TAGS_STATE.update(status="running", current=0, total=len(rels), photo="",
                      error="", name=name, hechas=0, salteadas=0, fallaron=0,
                      empezo=time.time(), stop=False, ultimos=[])
    threading.Thread(target=_tags_worker, args=(name, rels, load_config()),
                     daemon=True).start()
    return {"ok": True}


@app.route("/tags/lote/stop", methods=["POST"])
def tags_lote_stop():
    """Corta el etiquetado. Lo hecho hasta acá queda guardado."""
    TAGS_STATE["stop"] = True
    return {"ok": True}


@app.route("/tags/lote/estado")
def tags_lote_estado():
    """Para que la página se actualice sola sin recargar."""
    s = TAGS_STATE
    return {"status": s["status"], "current": s["current"], "total": s["total"],
            "photo": s["photo"], "name": s["name"], "hechas": s["hechas"],
            "salteadas": s["salteadas"], "fallaron": s["fallaron"],
            "error": s["error"],
            "lleva": _humano(time.time() - s["empezo"] if s["empezo"] else 0),
            "ultimos": s["ultimos"]}


@app.route("/post/<name>/caption", methods=["POST"])
def post_caption(name):
    """Escribe el pie de foto con el modelo de visión y lo guarda.

    LENTO: ~2 minutos en CPU. Por eso es a pedido (un botón) y queda guardado:
    la próxima vez que abras la foto sale al toque."""
    rel = request.form.get("rel", "")
    path = _photo_path(name, rel)
    if not path:
        return {"ok": False, "error": "Esa foto ya no está."}
    cfg = load_config()
    try:
        text = posts_mod.generate_caption(
            # el del desplegable del visor manda; si no eligieron, el de Ajustes;
            # y recién ahí el de fábrica (model=None)
            name, rel, str(path),
            model=request.form.get("model") or cfg.get("post_model") or None,
            crudeness=cfg.get("crudeness", posts_mod.CRUDENESS_DEFAULT),
            diminutivos=bool(cfg.get("diminutivos")))
    except Exception as e:
        return {"ok": False, "error": str(e)}
    return {"ok": True, "caption": text}


# Copete: un titular de noticia "tipo diario" para una foto, inventado por la IA
# que MIRA la foto (situación, ropa, contexto). El usuario lo pide con el
# botón 📰 del visor y queda guardado en data/copetes.json.
# El `post_model` de Ajustes es un modelo de TEXTO genérico (no mira la foto):
# para el copete se usa el VLM de visión abliterated de 7B, el mismo que
# escribe etiquetas y narración — mucho más específico que el genérico.
COPETE_MODEL = "hf.co/mradermacher/Qwen2.5-VL-7B-NSFW-Caption-V3-abliterated-GGUF:Q4_K_M"
COPETE_SYSTEM = """Sos un titularista de diario canchero. Mirás la foto con ganas
de titular, no de describir: elegís EL detalle que llama la atención (la pose,
la ropa, el gesto, el lugar, lo que está haciendo) y lo contás como noticia
CORTA y con gancho, en español, presente, de 4 a 8 palabras, en una sola línea.
Que pegue a ESTA foto, no un elogio genérico que le serviría a cualquiera. Sin
signos de exclamación, sin comillas, sin puntos intermedios ni al final, sin
subtítulos ni segunda frase, sin frases de relleno. Solo lo que se ve: no
inventes nada."""
COPETE_PROMPT = "Escribí el titular de diario de esta foto, en una sola línea, de 4 a 8 palabras, presente y con gancho."


def _copete_titular(t):
    """Deja el texto como titular corto: una sola línea, sin puntuación de
    cierre ni comillas, cortada a ~80 caracteres en un límite de palabra."""
    t = (t or "").strip().strip('"“”‘’').strip()
    t = t.split("\n")[0].strip()
    i = t.find(". ")
    if 0 < i < 70:
        t = t[:i].strip()
    t = t.rstrip(".").strip()
    if len(t) > 80:
        t = t[:80].rsplit(" ", 1)[0].rstrip(" .,").strip()
    return t


@app.route("/post/<name>/copete", methods=["POST"])
def post_copete(name):
    """Inventa un copete (titular de noticia) para la foto y lo guarda.

    Igual que el pie: LENTO en CPU (el modelo mira la foto), a pedido y queda
    guardado en data/copetes.json para no repetirlo."""
    rel = request.form.get("rel", "")
    path = _photo_path(name, rel)
    if not path:
        return {"ok": False, "error": "Esa foto ya no está."}
    cfg = load_config()
    try:
        texto = vlm.describe(
            str(path),
            model=request.form.get("model") or cfg.get("copete_model") or COPETE_MODEL,
            prompt=COPETE_PROMPT, system=COPETE_SYSTEM,
            num_predict=60, temperature=0.9)
    except Exception as e:
        return {"ok": False, "error": str(e)}
    texto = _copete_titular(texto)
    texto = copetes_mod.set_copete(name, rel, texto)
    return {"ok": True, "copete": texto}


# Visión (paso 1 del Centro de IA): la IA "mira" la foto y narra qué hay.
# Ese texto alimenta conceptualmente a etiquetas, pie y comentarios.
#
# POR QUÉ NARRACIÓN Y NO "FICHA OBJETIVA" (no volver atrás sin leer esto):
# antes pedía "una ficha objetiva de 2-4 líneas, sin opinar" y salía un
# inventario seco ("La imagen muestra a una mujer... Lleva lencería negra.").
# Lo que se quiere es narración: que el texto TENGA ritmo y detalle, que se
# pueda leer. Se cambió el registro, no la fidelidad — sigue mirando la foto y
# sigue prohibido inventar, que es lo único que hace confiable a este paso.
# Probado: en cuanto se le pide crudeza o "morbo", el modelo inventa desnudo
# que no está (ver el historial del narrador). La crudeza vive en el PIE, con
# su compuerta; acá el trabajo es MIRAR BIEN y contarlo bien.
#
# El "sin opinar" se mantiene a propósito: es una narración de lo que hay, no
# un comentario. Lo que se saca es "ficha"/"objetiva", que era lo que enfriaba
# el texto hasta volverlo un inventario.
VISION_SYSTEM = """Sos el ojo de una fototeca. Mirás la foto y la contás como
quien la está mirando de verdad: en español, con ritmo y detalle, quedándote en
lo que hay. Narración, no inventario: no una lista de datos, un texto que se
lea.

Narrás lo que se ve: la persona, la pose, la ropa (prenda por prenda, con color
y estado: puesta, abierta, corrida, levantada), el cuerpo, la piel a la vista,
el gesto, la mirada, el pelo, los tatuajes, el lugar, la luz.

No opinás ni calificás: nada de hermosa, sexy, provocante, sugerente. Tercera
persona, presente. Un párrafo de 4 a 7 frases, sin listas ni títulos.

Contás solo lo que ves. Si algo no se ve, no existe para vos."""
VISION_PROMPT = ("Contá lo que se ve en esta foto, con ritmo y detalle: la "
                 "persona, la pose, la ropa, el cuerpo, el gesto, el lugar.")


@app.route("/post/<name>/vision", methods=["POST"])
def post_vision(name):
    """La «visión» de la foto: la IA mira y anota qué hay. Es el paso 1 del
    Centro de IA; se guarda en posts.json como "vision"."""
    rel = request.form.get("rel", "")
    path = _photo_path(name, rel)
    if not path:
        return {"ok": False, "error": "Esa foto ya no está."}
    cfg = load_config()
    try:
        texto = vlm.describe(
            str(path),
            model=request.form.get("model") or cfg.get("post_model") or None,
            prompt=VISION_PROMPT, system=VISION_SYSTEM,
            num_predict=320, temperature=0.35)
    except Exception as e:
        return {"ok": False, "error": str(e)}
    texto = posts_mod.set_vision(name, rel, texto)
    return {"ok": True, "vision": texto}


@app.route("/post/<name>/fav", methods=["POST"])
def post_fav(name):
    """Marca/desmarca la foto como favorita (⭐)."""
    rel = request.form.get("rel", "")
    if not rel:
        return {"ok": False}
    return {"ok": True, "fav": posts_mod.set_fav(name, rel,
                                                 request.form.get("fav") == "1")}


@app.route("/post/<name>/people", methods=["POST"])
def post_people(name):
    """Guarda quiénes MÁS aparecen en la foto (etiquetado a mano con @).

    Se validan contra la base, así no se cuela un nombre que no existe."""
    rel = request.form.get("rel", "")
    if not rel:
        return {"ok": False}
    gente = [p for p in request.form.get("people", "").split(",") if p.strip()]
    return {"ok": True, "people": posts_mod.set_people(name, rel, gente)}


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
            crudeness=cfg.get("crudeness", posts_mod.CRUDENESS_DEFAULT),
            model=request.form.get("model") or cfg.get("post_model") or None,
            diminutivos=bool(cfg.get("diminutivos")),
            # "más comentarios" SUMA a los que ya están, como una publicación de
            # verdad; el 🗑 es el que borra
            sumar=request.form.get("sumar") == "1")
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


@app.route("/person/<name>/export")
def person_export(name):
    """Exporta TODO lo de una persona en un .zip: sus etiquetas, la voz, las
    fechas, la portada y los posts (pies + comentarios + quién aparece).

    NO mete las fotos originales: esas ya están en tu Biblioteca y pesan. Esto es
    el "perfil" —lo que el programa aprendió y vos cargaste— para respaldar o
    llevarte a otra máquina."""
    if name not in load_known():
        abort(404)
    library = get_library()
    fotos = (person_photos(library, name) +
             person_photos(library, name, only_nsfw=True)) if library else []
    perfil = {
        "nombre": name,
        "etiquetas_persona": tags_mod.get_person(name),
        "etiquetas_fotos": {r: tags_mod.get_photo(name, r)
                            for r in fotos if tags_mod.get_photo(name, r)},
        "voz": person_meta.get_voice(name),
        "meta": person_meta.get(name),
        "posts": posts_mod.get_many(name, fotos),
        "exportado": time.strftime("%Y-%m-%d %H:%M"),
    }
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(f"{safe_name(name)}/perfil.json",
                   json.dumps(perfil, ensure_ascii=False, indent=2))
        thumb = KNOWN_THUMBS / f"{safe_name(name)}.jpg"
        if thumb.is_file():
            z.write(thumb, f"{safe_name(name)}/portada.jpg")
    buf.seek(0)
    return Response(buf.getvalue(), mimetype="application/zip",
                    headers={"Content-Disposition":
                             f'attachment; filename="{safe_name(name)}_perfil.zip"'})


@app.route("/person/<name>/subir", methods=["POST"])
def person_subir(name):
    """Sube fotos directo al perfil: todas se guardan en la carpeta de esa
    persona, sin analizar caras (ya sabemos de quién son)."""
    files = request.files.getlist("fotos")
    if not files:
        return {"ok": False, "error": "No elegiste ninguna foto."}
    try:
        r = save_uploaded_photos(get_library(), name, files)
    except ValueError as e:
        return {"ok": False, "error": str(e)}
    return {"ok": True, **r}


@app.route("/tags/person/<name>", methods=["POST"])
def tags_person(name):
    """Guarda las etiquetas de una persona (texto libre, separadas por coma)."""
    tags = [t for t in request.form.get("tags", "").split(",")]
    saved = tags_mod.set_person(name, tags)
    return {"ok": True, "tags": saved}


@app.route("/tags/person/bulk", methods=["POST"])
def tags_person_bulk():
    """Suma (o saca) etiquetas a MUCHAS personas de una.

    El caso real: "estas 60 son hombres". Entrar a cada perfil no es una opción.

    Ojo con "add": SUMA. Si pisara la lista, etiquetar 60 personas de sexo les
    borraría todo lo que ya tenían."""
    names = [n for n in request.form.get("names", "").split("\n") if n.strip()]
    tags = [t for t in request.form.get("tags", "").split(",") if t.strip()]
    accion = request.form.get("action", "add")
    if not names or not tags:
        return {"ok": False, "error": "Faltan personas o etiquetas."}
    conocidos = load_known()
    out = {}
    for n in names:
        n = n.strip()
        if n not in conocidos:      # se borró mientras tanto
            continue
        tenia = tags_mod.get_person(n)
        if accion == "remove":
            quitar = {tags_mod._norm(t) for t in tags}
            nuevas = [t for t in tenia if tags_mod._norm(t) not in quitar]
        else:
            nuevas = tenia + tags   # set_person limpia duplicados y normaliza
        out[n] = tags_mod.set_person(n, nuevas)
    return {"ok": True, "n": len(out), "tags": out}


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
<h1>✎ Frases <span class="hint-ico" title="Las frases que escribe el descriptor son formularias y suenan robóticas («una imagen de una mujer acostada en una cama»). Acá las reescribís a tu manera, una sola vez: la próxima vez que una foto dé esa misma frase, se muestra la tuya. Como se repiten muchísimo, arreglás cientos de fotos con unas pocas ediciones.">ⓘ</span></h1>

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
    # salvaguarda: sin etiqueta de adulto el sugeridor NO describe cuerpo ni nada
    # sexual (igual que el pie y los comentarios). La duda cierra, no abre.
    sano = posts_mod.identidad_foto(name, rel)["sano"]
    # "sugerir más": las que ya salieron, para que el modelo tire otras
    evitar = [t for t in request.form.get("evitar", "").split(",") if t.strip()]
    try:
        r = interrogator.interrogate(str(path), vocab, engine=engine,
                                     vlm_model=request.form.get("vlm_model") or None,
                                     person=name, crudo=not sano, evitar=evitar)
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


@app.route("/tags/suggest_stream/<name>", methods=["POST"])
def tags_suggest_stream(name):
    """Igual que el sugeridor VLM pero en STREAMING: manda una etiqueta por línea
    (NDJSON) apenas el modelo la larga, para que aparezcan en vivo en vez de
    esperar toda la respuesta. La mirada del modelo se paga una sola vez."""
    library = get_library()
    rel = request.form.get("rel", "")
    path = Path(library) / name / rel if library else None
    if not path or not path.is_file():
        return {"ok": False, "error": "Esa foto ya no está."}, 404
    # misma salvaguarda que el resto: versión sana salvo adulto confirmado
    sano = posts_mod.identidad_foto(name, rel)["sano"]
    es_nsfw = NSFW_DIR_NAME in Path(rel).parts
    evitar = [t for t in request.form.get("evitar", "").split(",") if t.strip()]
    # el etiquetado usa el modelo que MEJOR mira (qwen2.5vl si está); el pie y los
    # comentarios siguen con el del desplegable. Si el form manda uno, gana.
    model = request.form.get("vlm_model") or vlm.tag_model()
    # "diccionario" de confianza para marcar (no borrar) palabras raras: lo que
    # el usuario YA usó en toda su base + la jerga cruda semilla. Una etiqueta de
    # una sola palabra que no está en ninguno = "nueva/dudosa" (posible invento).
    conocidas = {t.lower() for t in tags_mod.all_tags()} | vlm.VOCAB_CRUDO

    def gen():
        ev_local = list(evitar)
        # ACTO: clasificador dedicado (porntech/sex-position). Va PRIMERO y anclado.
        # Solo en fotos NSFW y solo con adulto confirmado. Nunca debe romper el
        # sugeridor.
        if es_nsfw and not sano and sexpos.disponible():
            try:
                acto = sexpos.clasificar(str(path))
                if acto and acto[0].lower() not in {e.strip().lower() for e in ev_local}:
                    ev_local.append(acto[0])     # que el VLM no lo repita
                    yield json.dumps({"tag": acto[0], "nueva": False,
                                      "acto": True}) + "\n"
            except Exception:
                pass
        n = 0
        try:
            for tag in vlm.suggest_tags_stream(str(path), model=model, quien=name,
                                               crudo=not sano, evitar=ev_local):
                n += 1
                nueva = len(tag.split()) == 1 and tag.lower() not in conocidas
                yield json.dumps({"tag": tag, "nueva": nueva}) + "\n"
        except Exception as e:
            yield json.dumps({"error": str(e)}) + "\n"
        else:
            # solo si NO hubo error: la foto pasó por el modelo (aunque n sea 0)
            procesadas.marcar(name, rel, n)

    return Response(stream_with_context(gen()), mimetype="application/x-ndjson")


PROBADOR_TEMPLATE = """
<style>
  .pbthumbs { display:flex; flex-wrap:wrap; gap:.4rem; margin-top:.5rem;
    max-height:14rem; overflow:auto; }
  .pbthumb-wrap { position:relative; }
  .pbthumb { width:74px; height:74px; border-radius:8px; object-fit:cover; cursor:pointer;
    border:2px solid transparent; }
  .pbthumb:hover { border-color:var(--acc); }
  .pbthumb.sel { border-color:var(--acc); }
  /* previa flotante en hover (fuera del contenedor con scroll, no se recorta) */
  .pbhover { position:fixed; z-index:60; width:240px; height:240px; object-fit:contain;
    border:2px solid var(--acc); border-radius:10px; box-shadow:0 8px 30px rgba(0,0,0,.6);
    background:#000; pointer-events:none; display:none; }
  .pbthumb-nsfw { position:absolute; top:2px; left:2px; font-size:.75rem;
    filter:drop-shadow(0 1px 2px rgba(0,0,0,.7)); pointer-events:none; }
  .pbthumb.marked { border-color:var(--acc); box-shadow:0 0 0 2px var(--acc) inset; }
  .pbthumb.viendo { outline:3px solid var(--acc2); outline-offset:1px; }
  .pbthumb-wrap.oculto { display:none; }              /* ocultar ya etiquetadas */
  .pbthumb-done { position:absolute; top:2px; right:2px; font-size:.7rem; color:#fff;
    background:#2e9c5b; border-radius:50%; width:1.05rem; height:1.05rem; line-height:1.05rem;
    text-align:center; pointer-events:none; }
  .pbthumb-ready { position:absolute; bottom:3px; right:3px; width:.55rem; height:.55rem;
    border-radius:50%; background:#4fd18b; box-shadow:0 0 0 1px rgba(0,0,0,.5); pointer-events:none; }
  .linklike2 { color:var(--acc2); cursor:pointer; text-decoration:underline; }
  #pb-preview { cursor:pointer; }
  .pbrow { display:grid; grid-template-columns:1fr 1rem minmax(0,1fr) 1.4rem;
    align-items:center; gap:.5rem; margin:.3rem 0; }
  .pbhead { display:grid; grid-template-columns:1fr 1rem minmax(0,1fr) 1.4rem;
    gap:.5rem; margin:.2rem 0 .4rem; font-size:.78rem; color:var(--mut);
    text-transform:uppercase; letter-spacing:.03em; }
  .pborig { color:var(--mut); font-size:.92rem; overflow:hidden; text-overflow:ellipsis;
    white-space:nowrap; }
  .pbarrow { color:var(--mut); text-align:center; }
  .pbrow input { background:var(--bg); border:1px solid var(--bd); border-radius:6px;
    padding:.4rem .6rem; color:var(--tx); font-size:.95rem; min-width:0; }
  .pbrow input.edit { border-color:var(--amb); }
  .pbrow .pbx { background:none; border:0; color:var(--mut); cursor:pointer; font-size:1rem; }
  .pbrow .pbx:hover { color:var(--red); }
  .pbrow.dropped .pborig { text-decoration:line-through; opacity:.5; }
  .pbrow.acto .pborig { color:#4fd18b; }
  .pbrow.acto input { border-color:#3a9d63; }
  /* revisión: lista a la izquierda, foto GRANDE pegada a la derecha */
  .pb-review { display:grid; grid-template-columns:1fr minmax(260px, 40%); gap:1rem; align-items:start; }
  .pb-review-foto { position:sticky; top:1rem; }
  .pb-review-foto img { width:100%; max-height:78vh; object-fit:contain; border-radius:10px;
    background:#000; cursor:pointer; display:block; }
  #pb-keys { margin-top:.55rem; opacity:.7; }
  #pb-keys kbd { background:var(--bg); border:1px solid var(--bd); border-radius:4px;
    padding:0 .35rem; font-size:.8em; }
  @media (max-width:820px) { .pb-review { grid-template-columns:1fr; } }
</style>
<h1>🧪 Probador de etiquetas <span class="hint-ico" title="Elegí una o varias fotos y etiquetalas de un saque: la IA va resolviendo de a una y, mientras revisás la primera, ya va preparando la siguiente. Corregís lo que está mal (ej: rusa → paja rusa) o descartás lo inventado (✕), guardás y pasás a la que sigue. Las correcciones quedan como diccionario para TODA la app.">ⓘ</span></h1>

<div class="panel">
  <div class="row">
    <label>Persona:</label>
    <select id="pb-persona" onchange="pbPersona()">
      <option value="">— elegí una persona —</option>
      {% for p in people %}<option value="{{ p }}">{{ p }}</option>{% endfor %}
    </select>
    <span class="meta">o</span>
    <label class="linklike2">subir un archivo<input type="file" id="pb-file" accept="image/*" hidden></label>
  </div>
  <div id="pb-thumbs" class="pbthumbs"></div>
  <div class="row" id="pb-selbar" style="display:none; margin-top:.4rem">
    <span class="meta"><b id="pb-seln">0</b> seleccionada(s)</span>
    <button type="button" class="secondary" onclick="pbSelTodas()">todas</button>
    <button type="button" class="secondary" onclick="pbSelNinguna()">ninguna</button>
    <button type="button" class="secondary" onclick="pbNsfwLote()" title="Marcar las seleccionadas como NSFW (se mueven a _nsfw)">🔞 NSFW en lote</button>
    <label class="meta" style="cursor:pointer"><input type="checkbox" id="pb-hide-done" onchange="pbAplicarOcultar()"> 🙈 ocultar ya etiquetadas (<b id="pb-done-n">0</b>✓)</label>
    <span class="meta">· Shift-clic = rango · Ctrl-clic = de a una</span>
  </div>
  <img id="pb-hover" class="pbhover" alt="">
  <img id="pb-preview" onclick="pbAbrirFull()" title="Clic para abrir la foto entera en otra pestaña"
       style="display:none; max-width:240px; border-radius:8px; margin-top:.6rem">
  <div class="row" style="margin-top:.6rem">
    <label>Modelo:</label>
    <select id="pb-model">
      {% for m in vlm_models %}<option value="{{ m }}" {% if m == tag_modelo %}selected{% endif %}>{% if m in vlm_crudos %}🔞 {% else %}👁 {% endif %}{{ m }}</option>{% endfor %}
    </select>
    <label style="font-weight:400"><input type="checkbox" id="pb-crudo" checked> crudo (NSFW)</label>
  </div>
  <div class="row" style="margin-top:.6rem">
    <button class="primary" id="pb-go" onclick="pbGo()" disabled>🔮 Etiquetar</button>
    <button class="secondary" id="pb-nsfw" onclick="pbNsfw()" disabled title="Mover esta foto a _nsfw">🔞 Marcar NSFW</button>
    <span class="meta" id="pb-st"></span>
  </div>
  <progress id="pb-prog" style="display:none; width:100%; margin-top:.5rem"></progress>
</div>

<div class="panel" id="pb-lista-panel" style="display:none">
  <div class="pb-review">
    <div class="pb-review-tags">
      <b>Etiquetas — comparás y corregís <span class="hint-ico" title="Izquierda = lo que vio la IA (no se toca) · derecha = nuestra interpretación (editala si está mal; queda como conversión para siempre) · ✕ descarta.">ⓘ</span></b>
      <div id="pb-empty" style="display:none; padding:.6rem .8rem; border:1px solid var(--amb);
           border-radius:8px; background:rgba(200,150,40,.08); margin:.4rem 0">
        <b id="pb-empty-msg">⚠ La IA no devolvió etiquetas para esta foto. <span class="hint-ico" title="Puede ser la foto (borrosa, sin nada que describir) o que el modelo se colgó en esta. Probá de nuevo, o saltala.">ⓘ</span></b>
        <button class="secondary" onclick="pbReintentar()">↻ Reintentar esta foto</button>
      </div>
      <div id="pb-listwrap">
        <div class="pbhead"><span>👁 vio la IA</span><span></span><span>✏️ interpretación</span><span></span></div>
        <div id="pb-lista"></div>
      </div>
      <div class="row" style="margin-top:.6rem">
        <button class="primary" id="pb-next" onclick="pbGuardarSiguiente()">💾 Guardar y siguiente ▶</button>
        <button class="secondary" id="pb-skip" onclick="pbSaltar()" title="Sin guardar, pasa a la próxima" style="display:none">⏭ Saltar</button>
        <span class="meta" id="pb-qcount"></span>
        <span class="meta" id="pb-save-st"></span>
      </div>
      <p class="meta" id="pb-keys"><kbd>Enter</kbd> guardar y siguiente · <kbd>S</kbd> saltar ·
        <kbd>R</kbd> reintentar · <kbd>←</kbd><kbd>→</kbd> navegar</p>
    </div>
    <div class="pb-review-foto">
      <img id="pb-bigphoto" onclick="pbAbrirFull()" title="Clic para abrir la foto entera en otra pestaña">
    </div>
  </div>
</div>

<script>
const pbg = id => document.getElementById(id);
let pbActual = null;      // foto en foco {tipo:'galeria'|'upload', name, rel, url, nsfw, file}
let pbFotos = [];         // [{name, rel, url, nsfw, thumb, el}] de la persona actual
let pbSel = new Set();    // índices seleccionados (para el lote)
let pbAnchor = null;      // último clic (para el rango con Shift)
let pbModo = 'galeria';   // 'galeria' | 'upload'

function pbPersona() {
  const name = pbg('pb-persona').value;
  const box = pbg('pb-thumbs'); box.innerHTML = '';
  pbFotos = []; pbSel.clear(); pbAnchor = null; pbModo = 'galeria';
  pbFinBatch();
  if (!name) { pbSelbar(); return; }
  fetch('/probador/fotos/' + encodeURIComponent(name)).then(r => r.json()).then(d => {
    let ndone = 0;
    (d.fotos || []).forEach((f, i) => {
      const w = document.createElement('span'); w.className = 'pbthumb-wrap';
      const im = document.createElement('img'); im.className = 'pbthumb'; im.src = f.thumb; im.title = f.rel;
      pbFotos.push({name: name, rel: f.rel, url: f.url, nsfw: f.nsfw, thumb: f.thumb, el: im, wrap: w, tagged: !!f.tagged});
      im.onclick = (e) => pbClickThumb(e, i);
      im.onmouseenter = () => pbHoverShow(im, f.url);
      im.onmouseleave = pbHoverHide;
      w.appendChild(im);
      if (f.nsfw) { const b = document.createElement('span'); b.className = 'pbthumb-nsfw'; b.textContent = '🔞'; w.appendChild(b); }
      if (f.tagged) {
        ndone++;
        const t = document.createElement('span'); t.className = 'pbthumb-done';
        // n_ia === 0: pasó por la IA y volvió VACÍA. Se marca distinto para poder
        // reintentarla a propósito en vez de darla por hecha.
        const vacia = f.n_ia === 0;
        t.textContent = vacia ? '∅' : '✓';
        if (vacia) t.style.background = 'var(--amb)';
        t.title = (vacia ? 'la IA no devolvió etiquetas' : 'ya pasó por la IA (' + f.n_ia + ')')
                  + (f.fecha_ia ? ' · ' + f.fecha_ia : '');
        w.appendChild(t);
      }
      box.appendChild(w);
    });
    pbg('pb-done-n').textContent = ndone;
    pbAplicarOcultar();
    pbSelbar();
  });
}
function pbFotoDe(rel) { return pbFotos.find(f => f.rel === rel); }
function pbAplicarOcultar() {          // 🙈 ocultar las que ya tienen etiquetas
  const ocultar = pbg('pb-hide-done').checked;
  pbFotos.forEach(f => f.wrap.classList.toggle('oculto', ocultar && f.tagged));
}
function pbMarcarEtiquetada(foto) {    // ✓ + saca de la vista si "ocultar" está activo
  if (!foto || foto.tagged) return;
  foto.tagged = true;
  const rd = foto.wrap.querySelector('.pbthumb-ready'); if (rd) rd.remove();
  if (!foto.wrap.querySelector('.pbthumb-done')) {
    const t = document.createElement('span'); t.className = 'pbthumb-done'; t.textContent = '✓';
    t.title = 'ya tiene etiquetas'; foto.wrap.appendChild(t);
  }
  pbg('pb-done-n').textContent = pbFotos.filter(x => x.tagged).length;
  if (pbg('pb-hide-done').checked) foto.wrap.classList.add('oculto');
}

// clic: normal = solo esa · Shift = rango desde el ancla · Ctrl/Cmd = suma de a una
function pbClickThumb(e, i) {
  if (e.shiftKey && pbAnchor !== null) {
    const a = Math.min(pbAnchor, i), b = Math.max(pbAnchor, i);
    for (let k = a; k <= b; k++) pbSel.add(k);
  } else if (e.ctrlKey || e.metaKey) {
    if (pbSel.has(i)) pbSel.delete(i); else pbSel.add(i);
    pbAnchor = i;
  } else {
    pbSel.clear(); pbSel.add(i); pbAnchor = i;
  }
  pbRepaint();
  pbPreview(i);                 // muestra la última clickeada
}
function pbSelTodas() { pbFotos.forEach((_, i) => pbSel.add(i)); pbRepaint(); }
function pbSelNinguna() { pbSel.clear(); pbRepaint(); }
function pbRepaint() {
  pbFotos.forEach((f, i) => f.el.classList.toggle('marked', pbSel.has(i)));
  pbSelbar();
}
function pbSelbar() {
  const n = pbSel.size;
  pbg('pb-selbar').style.display = pbFotos.length ? '' : 'none';
  pbg('pb-seln').textContent = n;
  const go = pbg('pb-go');
  go.disabled = (pbModo !== 'upload') && n === 0;
  go.textContent = pbModo === 'upload' ? '🔮 Etiquetar' : ('🔮 Etiquetar' + (n ? ' (' + n + ')' : ''));
}
function pbPreview(i) {          // pone la previa + botón NSFW de una foto
  const f = pbFotos[i]; if (!f) return;
  pbActual = {tipo: 'galeria', name: f.name, rel: f.rel, url: f.url, nsfw: f.nsfw};
  const img = pbg('pb-preview'); img.src = f.url; img.style.display = '';
  pbg('pb-nsfw').disabled = f.nsfw;
  pbg('pb-nsfw').textContent = f.nsfw ? '🔞 ya es NSFW' : '🔞 Marcar NSFW';
}
function pbAbrirFull() {         // la foto entera, de verdad, en otra pestaña (no modal)
  if (pbActual && pbActual.url) window.open(pbActual.url, '_blank');
}

pbg('pb-file').addEventListener('change', e => {
  const f = e.target.files[0];
  if (!f || !f.type.startsWith('image/')) return;
  pbg('pb-persona').value = ''; pbg('pb-thumbs').innerHTML = '';
  pbFotos = []; pbSel.clear(); pbAnchor = null; pbModo = 'upload';
  pbFinBatch();
  pbActual = {tipo: 'upload', file: f, url: URL.createObjectURL(f)};
  const img = pbg('pb-preview'); img.src = pbActual.url; img.style.display = '';
  pbg('pb-nsfw').disabled = true; pbg('pb-nsfw').textContent = '🔞 Marcar NSFW';
  pbSelbar();
});

const PB_RELOJ = ['⠋','⠙','⠹','⠸','⠼','⠴','⠦','⠧','⠇','⠏'];
const PB_ETQ_URL = "{{ url_for('probador_etiquetar') }}";
const PB_GRD_URL = "{{ url_for('probador_guardar') }}";
// estado del lote
let pbQueue = [], pbQi = 0, pbCache = new Map(), pbInflight = new Map(),
    pbShownRel = null, pbTimer = null;

function pbGo() {                       // botón Etiquetar: subida suelta o lote
  if (pbModo === 'upload') return pbEtiquetarUpload();
  const idx = [...pbSel].sort((a, b) => a - b);
  if (!idx.length) return;
  pbQueue = idx.map(i => pbFotos[i]);
  pbQi = 0; pbCache.clear(); pbInflight.clear();
  pbShow(0);
}

function pbLoading(on) {                // barra + spinner "mirando la foto… Ns"
  if (on) {
    pbg('pb-prog').style.display = '';
    let s = 0;
    const tick = () => { pbg('pb-st').textContent = ' ' + PB_RELOJ[s % 10] + ' mirando la foto… ' + s + 's'; s++; };
    tick(); pbTimer = setInterval(tick, 1000);
  } else {
    if (pbTimer) { clearInterval(pbTimer); pbTimer = null; }
    pbg('pb-prog').style.display = 'none'; pbg('pb-st').textContent = '';
  }
}

// pide las etiquetas de UNA foto; cachea el resultado y no repite pedidos en vuelo
function pbFetch(foto) {
  if (!foto) return Promise.resolve(null);
  if (pbCache.has(foto.rel)) return Promise.resolve(pbCache.get(foto.rel));
  if (pbInflight.has(foto.rel)) return pbInflight.get(foto.rel);
  const fd = new FormData();
  fd.append('model', pbg('pb-model').value);
  fd.append('crudo', pbg('pb-crudo').checked ? '1' : '0');
  fd.append('evitar', '[]');
  fd.append('name', foto.name); fd.append('rel', foto.rel);
  const p = fetch(PB_ETQ_URL, {method: 'POST', body: fd}).then(r => r.json())
    .then(d => { const res = d.ok ? {pares: d.pares || []} : {error: d.error || 'error'};
                 pbCache.set(foto.rel, res); pbInflight.delete(foto.rel);
                 // puntito verde = ya etiquetada y lista para revisar (solo si trajo algo)
                 if (res.pares && res.pares.length && foto.wrap && !foto.wrap.querySelector('.pbthumb-ready')) {
                   const dot = document.createElement('span'); dot.className = 'pbthumb-ready';
                   dot.title = 'ya etiquetada, lista para revisar'; foto.wrap.appendChild(dot);
                 }
                 return res; })
    .catch(e => { const res = {error: '' + e}; pbCache.set(foto.rel, res); pbInflight.delete(foto.rel); return res; });
  pbInflight.set(foto.rel, p);
  return p;
}

// muestra la foto i de la cola; cuando llega, dispara el etiquetado de la i+1
async function pbShow(i) {
  const foto = pbQueue[i]; if (!foto) return;
  pbQi = i;
  pbActual = {tipo: 'galeria', name: foto.name, rel: foto.rel, url: foto.url, nsfw: foto.nsfw};
  pbShownRel = foto.rel;
  const img = pbg('pb-preview'); img.src = foto.url; img.style.display = '';
  pbg('pb-bigphoto').src = foto.url;             // foto grande al lado de la lista
  pbg('pb-nsfw').disabled = foto.nsfw;
  pbg('pb-nsfw').textContent = foto.nsfw ? '🔞 ya es NSFW' : '🔞 Marcar NSFW';
  pbFotos.forEach(f => f.el.classList.toggle('viendo', f.rel === foto.rel));
  pbCounter();
  pbg('pb-lista-panel').style.display = ''; pbg('pb-lista').innerHTML = '';
  pbOcultarVacio();
  pbLoading(true);
  const res = await pbFetch(foto);
  if (pbShownRel !== foto.rel) return;          // el usuario ya saltó a otra
  pbLoading(false);
  if (res && res.error) pbMostrarVacio(res.error, true);
  else pbLlenar(res ? res.pares : []);
  if (pbQueue[i + 1]) pbFetch(pbQueue[i + 1]);   // ADELANTO la próxima (lookahead 1)
}

function pbMostrarVacio(msg, esError) {          // caja clara cuando no vino nada / error
  pbg('pb-empty').style.display = '';
  pbg('pb-listwrap').style.display = 'none';
  pbg('pb-empty-msg').textContent = (esError ? '⛔ Error: ' : '⚠ ') + msg;
  pbg('pb-empty').style.borderColor = esError ? 'var(--red)' : 'var(--amb)';
  pbg('pb-st').textContent = '';
}
function pbOcultarVacio() {
  pbg('pb-empty').style.display = 'none';
  pbg('pb-listwrap').style.display = '';
}
function pbReintentar() {                         // vuelve a pedir ESTA foto (olvida el vacío)
  if (pbModo === 'upload') { pbEtiquetarUpload(); return; }
  const foto = pbQueue[pbQi]; if (!foto) return;
  pbCache.delete(foto.rel); pbInflight.delete(foto.rel);
  pbShow(pbQi);
}

function pbCounter() {
  const total = pbQueue.length, cur = pbQi + 1, last = pbQi >= total - 1;
  pbg('pb-qcount').textContent = total > 1 ? ('foto ' + cur + '/' + total) : '';
  pbg('pb-next').textContent = last ? '💾 Guardar y terminar' : '💾 Guardar y siguiente ▶';
  pbg('pb-skip').style.display = total > 1 ? '' : 'none';
}

function pbLlenar(pares) {
  pbg('pb-lista-panel').style.display = '';
  if (!pares || !pares.length) { pbMostrarVacio('La IA no devolvió etiquetas para esta foto.', false); return; }
  pbOcultarVacio();
  const lista = pbg('pb-lista'); lista.innerHTML = '';
  pares.forEach(p => pbFila(p.orig, p.final, p.acto));
}

function pbEtiquetarUpload() {          // subida suelta: una sola, sin cola
  if (!pbActual || pbActual.tipo !== 'upload') return;
  pbQueue = []; pbShownRel = null;
  pbCounter();
  pbg('pb-lista-panel').style.display = ''; pbg('pb-lista').innerHTML = '';
  pbg('pb-bigphoto').src = pbActual.url || '';
  pbOcultarVacio();
  pbLoading(true);
  const fd = new FormData();
  fd.append('model', pbg('pb-model').value);
  fd.append('crudo', pbg('pb-crudo').checked ? '1' : '0');
  fd.append('evitar', '[]');
  fd.append('foto', pbActual.file);
  fetch(PB_ETQ_URL, {method: 'POST', body: fd}).then(r => r.json()).then(d => {
    pbLoading(false);
    if (!d.ok) { pbg('pb-st').textContent = ' ' + (d.error || 'error'); return; }
    pbLlenar(d.pares || []);
  }).catch(e => { pbLoading(false); pbg('pb-st').textContent = ' ' + e; });
}
// orig = lo que dijo la IA (izq, no editable) · final = interpretación (der, editable)
// acto = viene del detector de actos dedicado (sexpos), va marcado 🎯 en verde
function pbFila(orig, final, acto) {
  const row = document.createElement('div'); row.className = 'pbrow' + (acto ? ' acto' : '');
  const izq = document.createElement('span'); izq.className = 'pborig';
  izq.textContent = orig; izq.title = orig;
  const flecha = document.createElement('span'); flecha.className = 'pbarrow'; flecha.textContent = '→';
  const inp = document.createElement('input'); inp.value = final; inp.dataset.orig = final;
  inp.oninput = () => inp.classList.toggle('edit', inp.value.trim() !== inp.dataset.orig);
  const x = document.createElement('button'); x.className = 'pbx'; x.textContent = '✕'; x.title = 'Descartar';
  x.onclick = () => row.remove();
  row.appendChild(izq); row.appendChild(flecha); row.appendChild(inp); row.appendChild(x);
  pbg('pb-lista').appendChild(row);
}

function pbNsfw() {
  if (!pbActual || pbActual.tipo !== 'galeria' || pbActual.nsfw) return;
  if (!confirm('¿Marcar esta foto como NSFW? Se mueve a _nsfw.')) return;
  const oldRel = pbActual.rel;
  fetch('/post/' + encodeURIComponent(pbActual.name) + '/nsfw', {method: 'POST',
    headers: {'Content-Type': 'application/x-www-form-urlencoded'},
    body: new URLSearchParams({rel: oldRel})}).then(r => r.json()).then(d => {
      if (!d.ok) { alert(d.error || 'No se pudo.'); return; }
      pbActual.rel = d.new_rel; pbActual.url = d.new_url; pbActual.nsfw = true;
      // CLAVE: la foto se MOVIÓ a _nsfw, así que hay que actualizar su ruta también
      // en pbFotos (de ahí sale la ruta al Etiquetar) y olvidar la cacheada vieja.
      const foto = pbFotoDe(oldRel);
      if (foto) {
        foto.rel = d.new_rel; foto.url = d.new_url; foto.nsfw = true;
        foto.el.title = d.new_rel; foto.el.src = d.new_url;
        if (!foto.wrap.querySelector('.pbthumb-nsfw')) {
          const b = document.createElement('span'); b.className = 'pbthumb-nsfw'; b.textContent = '🔞';
          foto.wrap.appendChild(b);
        }
      }
      pbCache.delete(oldRel); pbInflight.delete(oldRel);
      if (pbShownRel === oldRel) pbShownRel = d.new_rel;
      pbg('pb-preview').src = d.new_url;
      pbg('pb-nsfw').disabled = true; pbg('pb-nsfw').textContent = '🔞 ya es NSFW';
      pbg('pb-st').textContent = ' ✓ marcada NSFW';
      if (window.refrescarUndo) window.refrescarUndo();
      setTimeout(() => pbg('pb-st').textContent = '', 2000);
    });
}

async function pbNsfwLote() {           // marcar todas las SELECCIONADAS como NSFW
  const targets = [...pbSel].map(i => pbFotos[i]).filter(f => f && !f.nsfw);
  if (!targets.length) {
    pbg('pb-st').textContent = ' no hay seleccionadas sin NSFW';
    setTimeout(() => pbg('pb-st').textContent = '', 2500); return;
  }
  if (!confirm('¿Marcar ' + targets.length + ' foto(s) como NSFW? Se mueven a _nsfw.')) return;
  let ok = 0;
  for (const foto of targets) {
    const oldRel = foto.rel;
    try {
      const r = await fetch('/post/' + encodeURIComponent(foto.name) + '/nsfw', {method: 'POST',
        headers: {'Content-Type': 'application/x-www-form-urlencoded'},
        body: new URLSearchParams({rel: oldRel})});
      const d = await r.json();
      if (d.ok) {
        pbCache.delete(oldRel); pbInflight.delete(oldRel);
        foto.rel = d.new_rel; foto.url = d.new_url; foto.nsfw = true;
        foto.el.title = d.new_rel; foto.el.src = d.new_url;
        if (!foto.wrap.querySelector('.pbthumb-nsfw')) {
          const b = document.createElement('span'); b.className = 'pbthumb-nsfw'; b.textContent = '🔞';
          foto.wrap.appendChild(b);
        }
        ok++;
        pbg('pb-st').textContent = ' 🔞 marcando… ' + ok + '/' + targets.length;
      }
    } catch (e) { /* seguir con las demás */ }
  }
  pbg('pb-st').textContent = ' ✓ ' + ok + ' marcada(s) NSFW';
  setTimeout(() => pbg('pb-st').textContent = '', 3000);
  pbSelbar();
  if (window.refrescarUndo) window.refrescarUndo();
}

function pbGuardarActual(despues) {     // guarda la lista de la foto en foco
  const pares = {}, tags = [];
  document.querySelectorAll('#pb-lista .pbrow input').forEach(inp => {
    const v = inp.value.trim(), o = (inp.dataset.orig || '').trim();
    if (!v) return; tags.push(v);
    if (v.toLowerCase() !== o.toLowerCase()) pares[o] = v;   // corregida -> conversión
  });
  const relGuardada = (pbActual && pbActual.tipo === 'galeria') ? pbActual.rel : null;
  const tuvo = tags.length > 0;                 // solo marcamos ✓ si de verdad guardó algo
  pbg('pb-save-st').textContent = ' guardando…';
  const body = new URLSearchParams({pares: JSON.stringify(pares), tags: JSON.stringify(tags)});
  if (relGuardada) { body.append('name', pbActual.name); body.append('rel', relGuardada); }
  fetch(PB_GRD_URL, {method: 'POST',
    headers: {'Content-Type': 'application/x-www-form-urlencoded'}, body: body})
    .then(r => r.json()).then(d => {
      pbg('pb-save-st').textContent = d.ok
        ? (' ✓ ' + d.aplicadas + ' etiqueta(s) · ' + d.conversiones + ' conversión(es)') : ' error';
      setTimeout(() => pbg('pb-save-st').textContent = '', 3000);
      if (d.ok && tuvo && relGuardada) pbMarcarEtiquetada(pbFotoDe(relGuardada));
      if (d.ok && window.refrescarUndo) window.refrescarUndo();
      if (despues) despues();
    })
    .catch(e => { pbg('pb-save-st').textContent = ' ' + e; });
}
function pbGuardarSiguiente() { pbGuardarActual(pbAvanzar); }
function pbSaltar() { pbAvanzar(); }
function pbAvanzar() {                   // pasa a la próxima de la cola (o termina)
  if (pbModo === 'upload' || pbQueue.length <= 1) { pbFinBatch(); return; }
  if (pbQi + 1 < pbQueue.length) pbShow(pbQi + 1);
  else pbFinBatch(pbQueue.length);
}
function pbFinBatch(n) {                 // cierra el lote y limpia el resaltado
  pbQueue = []; pbShownRel = null;
  pbg('pb-lista-panel').style.display = 'none';
  pbg('pb-qcount').textContent = '';
  pbFotos.forEach(f => f.el.classList.remove('viendo'));
  if (n) { pbg('pb-save-st').textContent = ' ✓ listo, ' + n + ' foto(s) revisada(s)';
           setTimeout(() => pbg('pb-save-st').textContent = '', 4000); }
}

/* ---- previa flotante al pasar el mouse por una miniatura ---- */
function pbHoverShow(el, url) {
  const h = pbg('pb-hover'); h.src = url;
  const r = el.getBoundingClientRect(), S = 240, gap = 10;
  let left = r.right + gap;
  if (left + S > window.innerWidth) left = r.left - gap - S;   // no entra: al otro lado
  left = Math.max(4, left);
  let top = r.top + r.height / 2 - S / 2;
  top = Math.max(4, Math.min(top, window.innerHeight - S - 4));
  h.style.left = left + 'px'; h.style.top = top + 'px'; h.style.display = 'block';
}
function pbHoverHide() { pbg('pb-hover').style.display = 'none'; }

/* ---- atajos de teclado (solo mientras revisás una foto) ---- */
function pbNavegar(delta) {              // ←/→ : moverse por la cola sin guardar
  if (pbModo === 'upload' || !pbQueue.length) return;
  const j = pbQi + delta;
  if (j >= 0 && j < pbQueue.length) pbShow(j);
}
window.addEventListener('keydown', e => {
  if (getComputedStyle(pbg('pb-lista-panel')).display === 'none') return;   // no hay revisión abierta
  const ae = document.activeElement;
  const typing = ae && (ae.tagName === 'INPUT' || ae.tagName === 'TEXTAREA');
  if (e.key === 'Enter') {
    e.preventDefault();
    if (typing) { ae.blur(); return; }   // dentro de un campo: confirma el campo, NO avanza
    pbGuardarSiguiente();                 // fuera de un campo: guardar y siguiente
    return;
  }
  if (typing) return;                     // escribiendo: dejar tipear letras/flechas normal
  const k = e.key.toLowerCase();
  if (k === 's') { e.preventDefault(); pbSaltar(); }
  else if (k === 'r') { e.preventDefault(); pbReintentar(); }
  else if (e.key === 'ArrowRight') { e.preventDefault(); pbNavegar(1); }
  else if (e.key === 'ArrowLeft') { e.preventDefault(); pbNavegar(-1); }
});
</script>
"""


@app.route("/probador")
def probador():
    """Probador: elegís una foto (galería o subida), la IA la etiqueta, corregís/
    descartás en una lista, y al guardar las correcciones quedan como diccionario
    y las etiquetas van a la foto."""
    # las personas: las de la base de caras MÁS las importadas por carpeta (que
    # no tienen cara aprendida). Igual que la Galería, para que no falte ninguna.
    personas = sorted(set(load_known()) | set(library_people(get_library())),
                      key=str.lower)
    return render_template_string(
        shell("probador", PROBADOR_TEMPLATE),
        vlm_models=vlm.list_models(), tag_modelo=vlm.tag_model(),
        vlm_crudos=vlm.modelos_descensurados(), people=personas)


@app.route("/probador/fotos/<name>")
def probador_fotos(name):
    """Las fotos de una persona (normales + nsfw) para el selector del probador."""
    library = get_library()
    if not library:
        return {"ok": False, "fotos": []}
    fotos = []
    # el ✓ marca "ya pasó por el modelo", no "tiene etiquetas": una puesta a mano
    # no cuenta, y una que el modelo devolvió vacía SÍ (para no repetirla)
    hechas = procesadas.de_persona(name)
    for rel in person_photos(library, name) + person_photos(library, name, only_nsfw=True):
        marca = hechas.get(rel)
        fotos.append({"rel": rel, "nsfw": NSFW_DIR_NAME in Path(rel).parts,
                      "tagged": marca is not None,
                      # 0 = pasó por la IA pero no devolvió nada (se puede reintentar)
                      "n_ia": (marca or {}).get("n"),
                      "fecha_ia": (marca or {}).get("fecha"),
                      "thumb": url_for("person_thumb", name=name, rel=rel),
                      "url": url_for("person_photo", name=name, rel=rel)})
    return {"ok": True, "fotos": fotos}


def _modo_sano(name):
    """True = modo seguro: nada de cuerpo ni sexual. Solo da False con la persona
    confirmada adulta por etiqueta (ver posts.identidad).

    Sin nombre, o si algo falla leyendo las etiquetas, devuelve True. Antes era al
    revés y eso era el agujero: cualquier error dejaba la salvaguarda apagada."""
    if not name:
        return True
    try:
        tags = tags_mod.get_person(name) + tags_mod.get_photo(name, "")
        return posts_mod.identidad(tags)["sano"]
    except Exception:
        return True


def _acto_pares(path, crudo, name, pares, evitar):
    """Si crudo (NSFW) y la persona está CONFIRMADA adulta, corre el detector de
    actos (sexpos) y devuelve el acto anclado 🎯 como par para PONER PRIMERO, salvo
    que ya esté entre las etiquetas del VLM o entre las evitadas. Lista vacía si no
    aplica. El acto viene de un modelo dedicado, no del VLM: por eso va marcado."""
    if not crudo or _modo_sano(name) or not sexpos.disponible():
        return []
    try:
        acto = sexpos.clasificar(path)         # (etiqueta_es, confianza) o None
    except Exception:
        acto = None
    if not acto:
        return []
    palabra, conf = acto
    k = vlm._dup_key(palabra.lower())
    ya = {vlm._dup_key(p["final"].lower()) for p in pares}
    ya |= {vlm._dup_key(str(e).strip().lower()) for e in evitar}
    if k in ya:
        return []
    return [{"orig": "detector de actos 🎯 (" + str(round(conf * 100)) + "%)",
             "final": palabra, "acto": True}]


@app.route("/probador/etiquetar", methods=["POST"])
def probador_etiquetar():
    """Etiqueta una foto (de galería por name+rel, o subida) y devuelve la lista
    final para que el usuario la corrija."""
    model = request.form.get("model") or vlm.tag_model()
    crudo = request.form.get("crudo") == "1"
    name = request.form.get("name", "")
    rel = request.form.get("rel", "")
    try:
        evitar = json.loads(request.form.get("evitar", "[]"))
        evitar = [str(e) for e in evitar if str(e).strip()]
    except Exception:
        evitar = []
    if name and rel:
        library = get_library()
        path = Path(library) / name / rel if library else None
        if not path or not path.is_file():
            return {"ok": False, "error": "Esa foto ya no está."}
        try:
            # vlm.suggest_tags_pairs devuelve una lista de {orig, final} (dos columnas)
            # si el modelo se colga en medio, el _acto_pares rescata lo que queda
            pares = vlm.suggest_tags_pairs(str(path), model=model, quien=name,
                                           crudo=crudo, evitar=evitar)
            # _acto_pares añade el 🎯 de acto porntech al inicio (si es NSFW y adulto)
            pares = _acto_pares(str(path), crudo, name, pares, evitar) + pares
            # queda anotado que ESTA foto ya pasó por el modelo, aunque haya
            # vuelto con 0 etiquetas (ese caso antes se perdía y la repetías)
            procesadas.marcar(name, rel, len(pares))
            return {"ok": True, "pares": pares or []}
        except Exception as e:
            return {"ok": False, "error": str(e)}
    f = request.files.get("foto")
    if not f:
        return {"ok": False, "error": "Falta la foto."}
    import tempfile
    suf = Path(f.filename or "x.jpg").suffix or ".jpg"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suf)
    try:
        f.save(tmp.name)
        tmp.close()
        pares = vlm.suggest_tags_pairs(tmp.name, model=model, crudo=crudo, evitar=evitar)
        pares = _acto_pares(tmp.name, crudo, "", pares, evitar) + pares
        return {"ok": True, "pares": pares}
    except Exception as e:
        return {"ok": False, "error": str(e)}
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass


@app.route("/probador/guardar", methods=["POST"])
def probador_guardar():
    """Guarda las correcciones como conversiones (rusa->paja rusa, para TODA la
    app) y aplica las etiquetas a la foto de galería (si vino name+rel)."""
    try:
        pares = json.loads(request.form.get("pares", "{}"))
    except Exception:
        pares = {}
    cfg = load_config()
    conv = dict(cfg.get("f_conversiones", {}) or {})
    for k, v in pares.items():
        k, v = str(k).strip(), str(v).strip()
        if k and v and k.lower() != v.lower():
            conv[k] = v
    cfg["f_conversiones"] = conv
    save_config(cfg)
    aplicadas = 0
    name = request.form.get("name", "")
    rel = request.form.get("rel", "")
    if name and rel:
        try:
            tags = [str(t).strip() for t in json.loads(request.form.get("tags", "[]"))
                    if str(t).strip()]
            propias = tags_mod.get_photo(name, rel)
            nuevas = propias + [t for t in tags if t not in propias]
            if nuevas != propias:        # solo si cambió algo, para poder deshacer
                _undo_push({"tipo": "tags", "desc": f"guardar etiquetas «{Path(rel).name}»",
                            "name": name, "rel": rel, "prev": propias})
            tags_mod.set_photo(name, rel, nuevas)
            aplicadas = len(tags)
        except Exception:
            pass
    return {"ok": True, "conversiones": len(conv), "aplicadas": aplicadas}


CONVERSIONES_TEMPLATE = """
<style>
  .cv-head, .cv-row { display:grid; grid-template-columns:1fr 1.2rem minmax(0,1fr) 1.6rem;
    gap:.5rem; align-items:center; }
  .cv-head { font-size:.78rem; color:var(--mut); text-transform:uppercase;
    letter-spacing:.03em; margin:.2rem 0 .4rem; }
  .cv-row { margin:.3rem 0; }
  .cv-row input { background:var(--bg); border:1px solid var(--bd); border-radius:6px;
    padding:.4rem .6rem; color:var(--tx); font-size:.95rem; min-width:0; }
  .cv-row input.edit { border-color:var(--amb); }
  .cv-arrow { color:var(--mut); text-align:center; }
  .cv-x { background:none; border:0; color:var(--mut); cursor:pointer; font-size:1rem; }
  .cv-x:hover { color:var(--red); }
</style>
<h1>🔁 Diccionario de conversiones <span class="hint-ico" title="Cuando el modelo dice la palabra de la izquierda, en TODA la app se reemplaza por la de la derecha (ej: rusa → paja rusa). Se crean solas desde el Probador; acá las ves, editás y borrás.">ⓘ</span></h1>
{% if message %}<div class="msg">{{ message }}</div>{% endif %}

{% if parecidas %}
<div class="panel" style="border-color:var(--amb)">
  <b>🧹 Etiquetas casi iguales <span class="hint-ico" title="La misma cosa escrita distinto (denuda/desnuda). Cada variante es un molde que no dispara y un alias que tenés que escribir a mano. Al juntarlas se conserva la MÁS usada y se reemplaza en las fotos de todas las personas.">ⓘ</span></b>
  <div id="pa-lista" style="margin-top:.5rem">
    {% for a, na, b, nb, r in parecidas %}
    <div class="row pa-fila" style="margin:.25rem 0; gap:.5rem">
      <span class="chip">{{ b }} <b class="meta">{{ nb }}</b></span>
      <span class="meta">→</span>
      <span class="chip">{{ a }} <b class="meta">{{ na }}</b></span>
      <button class="secondary sm" onclick="paJuntar(this)"
              data-vieja="{{ b|e }}" data-nueva="{{ a|e }}">juntar</button>
      <button class="secondary sm" onclick="this.closest('.pa-fila').remove()"
              title="Son cosas distintas, no las junto">✕</button>
    </div>
    {% endfor %}
  </div>
  <span class="meta" id="pa-st"></span>
</div>
{% endif %}

<div class="panel">
  <div class="row" style="margin-bottom:.5rem">
    <input id="cv-filtro" class="path" placeholder="🔍 filtrar…" oninput="cvFiltrar()" autofocus>
    <span class="meta"><b id="cv-count">{{ pares|length }}</b> conversión(es)</span>
  </div>
  <div class="cv-head" {% if not pares %}style="display:none" id="cv-head"{% else %}id="cv-head"{% endif %}>
    <span>dice la IA</span><span></span><span>lo reemplazamos por</span><span></span>
  </div>
  <div id="cv-lista">
    {% for k, v in pares %}
    <div class="cv-row">
      <input class="cv-orig" value="{{ k }}" spellcheck="false">
      <span class="cv-arrow">→</span>
      <input class="cv-final" value="{{ v }}" spellcheck="false">
      <button class="cv-x" title="Borrar" onclick="this.closest('.cv-row').remove(); cvContar()">✕</button>
    </div>
    {% endfor %}
  </div>
  <p class="meta">Todavía no hay ninguna conversión. Se van creando cuando corregís una etiqueta en el Probador.</p>
  <div class="row" style="margin-top:.7rem">
    <button class="secondary" onclick="cvAgregar()">＋ agregar una</button>
    <button class="primary" onclick="cvGuardar()">💾 Guardar cambios</button>
    <span class="meta" id="cv-st"></span>
  </div>
</div>

<script>
const cvg = id => document.getElementById(id);
function paJuntar(btn) {
  const vieja = btn.dataset.vieja, nueva = btn.dataset.nueva;
  if (!confirm('Reemplazar «' + vieja + '» por «' + nueva + '» en las fotos de todas las personas. ¿Seguro?')) return;
  cvg('pa-st').textContent = ' juntando…';
  fetch("{{ url_for('tags_merge') }}", {method: 'POST',
    headers: {'Content-Type': 'application/x-www-form-urlencoded'},
    body: new URLSearchParams({vieja: vieja, nueva: nueva})})
    .then(r => r.json()).then(d => {
      if (!d.ok) { cvg('pa-st').textContent = ' ✗ ' + (d.error || 'error'); return; }
      btn.closest('.pa-fila').remove();
      cvg('pa-st').textContent = ' ✓ ' + d.n + ' cambiadas';
    }).catch(e => cvg('pa-st').textContent = ' ' + e);
}
function cvContar() {
  const n = document.querySelectorAll('#cv-lista .cv-row').length;
  cvg('cv-count').textContent = n;
  cvg('cv-head').style.display = n ? '' : 'none';
  cvg('cv-vacio').style.display = n ? 'none' : '';
}
function cvFiltrar() {
  const q = cvg('cv-filtro').value.trim().toLowerCase();
  document.querySelectorAll('#cv-lista .cv-row').forEach(row => {
    const a = row.querySelector('.cv-orig').value.toLowerCase();
    const b = row.querySelector('.cv-final').value.toLowerCase();
    row.style.display = (!q || a.includes(q) || b.includes(q)) ? '' : 'none';
  });
}
function cvAgregar() {
  const row = document.createElement('div'); row.className = 'cv-row';
  const a = document.createElement('input'); a.className = 'cv-orig'; a.placeholder = 'lo que dice la IA'; a.spellcheck = false;
  const fl = document.createElement('span'); fl.className = 'cv-arrow'; fl.textContent = '→';
  const b = document.createElement('input'); b.className = 'cv-final'; b.placeholder = 'lo que querés'; b.spellcheck = false;
  const x = document.createElement('button'); x.className = 'cv-x'; x.textContent = '✕'; x.title = 'Borrar';
  x.onclick = () => { row.remove(); cvContar(); };
  row.appendChild(a); row.appendChild(fl); row.appendChild(b); row.appendChild(x);
  cvg('cv-lista').appendChild(row); cvContar(); a.focus();
}
function cvGuardar() {
  const pares = [];
  document.querySelectorAll('#cv-lista .cv-row').forEach(row => {
    const orig = row.querySelector('.cv-orig').value.trim();
    const final = row.querySelector('.cv-final').value.trim();
    if (orig && final) pares.push({orig: orig, final: final});
  });
  cvg('cv-st').textContent = ' guardando…';
  fetch("{{ url_for('conversiones_save') }}", {method: 'POST',
    headers: {'Content-Type': 'application/x-www-form-urlencoded'},
    body: new URLSearchParams({pares: JSON.stringify(pares)})})
    .then(r => r.json()).then(d => {
      cvg('cv-st').textContent = d.ok ? (' ✓ guardadas ' + d.n) : ' error';
      setTimeout(() => cvg('cv-st').textContent = '', 3000);
    }).catch(e => cvg('cv-st').textContent = ' ' + e);
}
</script>
"""


MOLDES_TEMPLATE = """
<style>
  .mo-cuerpo { display:grid; grid-template-columns:minmax(0,1.1fr) minmax(0,1.4fr) auto 1.6rem;
    gap:.5rem; align-items:start; margin:.4rem 0; }
  .mo-cuerpo input, .mo-cuerpo textarea { background:var(--bg); border:1px solid var(--bd);
    border-radius:6px; padding:.4rem .6rem; color:var(--tx); font-size:.95rem; min-width:0;
    font-family:inherit; }
  .mo-cuerpo textarea { resize:vertical; min-height:2.4rem; line-height:1.35; }
  .mo-cuerpo input.tags { border-color:var(--acc2); }
  .mo-head { display:grid; grid-template-columns:minmax(0,1.1fr) minmax(0,1.4fr) auto 1.6rem;
    gap:.5rem; font-size:.78rem; color:var(--mut); text-transform:uppercase;
    letter-spacing:.03em; margin:.2rem 0 .4rem; }
  .mo-x { background:none; border:0; color:var(--mut); cursor:pointer; font-size:1rem; }
  .mo-x:hover { color:var(--red); }
  .mo-nsfw { display:flex; flex-direction:column; align-items:center; gap:.15rem; font-size:.8rem;
    color:var(--mut); white-space:nowrap; cursor:pointer; }
  .mo-bdg { background:var(--amb); color:#000; border-radius:4px; padding:0 .3rem; font-size:.68rem; }
  .mo-cuerpo.es-borrador { border-left:3px solid var(--amb); padding-left:.5rem; }
  .mo-hueco { display:inline-block; background:var(--bg); border:1px solid var(--amb);
    border-radius:20px; padding:.2rem .6rem; margin:.15rem; font-size:.85rem; cursor:pointer; }
  .mo-hueco:hover { border-color:var(--acc); color:var(--acc); }
  .mo-hueco b { color:var(--amb); }
  details.mo-sec { margin:.8rem 0; }
  details.mo-sec > summary { cursor:pointer; font-weight:600; padding:.3rem 0; }
</style>
<h1>✍ Pies de foto <span class="hint-ico" title="El pie lo arman tus moldes, al instante, sin IA (probamos seis modelos y ninguno sonaba natural). Un molde se dispara cuando la foto tiene todas sus etiquetas. Escribí en rioplatense, en primera persona («no me quiero ir más»). Los de 🔞 solo salen con la persona marcada adulto. Truco de género: poné @ donde cambia (pelirroj@, pervertid@) y prende con las dos formas. Sin @ no se mezcla nada (manzano ≠ manzana).">ⓘ</span></h1>
{% if message %}<div class="msg">{{ message }}</div>{% endif %}

{% if cob %}
<div class="panel">
  <b>📊 Cobertura de pies <span class="hint-ico" title="Cuántas de tus fotos reciben un pie de molde (instantáneo) y cuántas caen al modelo (lento y narrado). Abajo, las combinaciones que más fotos destraban si les escribís un molde.">ⓘ</span></b>
  <div class="row" style="margin:.4rem 0; gap:1.2rem">
    <span><b style="font-size:1.3rem">{{ cob.con_molde }}</b> <span class="meta">con molde</span></span>
    <span><b style="font-size:1.3rem; color:var(--amb)">{{ cob.sin_molde }}</b> <span class="meta">caen al modelo</span></span>
    <span><b style="font-size:1.3rem; color:var(--mut)">{{ cob.sin_tags }}</b> <span class="meta">sin etiquetar</span></span>
  </div>
  {% if cob.faltan %}
  <p class="meta" style="margin:.3rem 0 .2rem">Combinaciones que más te rinde escribir (fotos que destraba cada una):</p>
  <div>
    {% for combo, n in cob.faltan %}
    <span class="mo-hueco" onclick="moDesdeCombo(this)" data-combo="{{ combo|e }}">{{ combo }} <b>{{ n }}</b></span>
    {% endfor %}
  </div>
  {% endif %}
</div>
{% endif %}

{% if huecos %}
<div class="panel" style="border-color:var(--amb)">
  <b>⚠ Etiquetas tuyas que todavía no tienen molde <span class="hint-ico" title="Aparecen en tus fotos pero no disparan ningún pie, así que esas fotos caen al modelo (lento y narrado). Tocá una para empezar un molde con esa etiqueta. Ordenadas por las que más se repiten.">ⓘ</span></b>
  <div style="margin-top:.4rem">
    {% for tag, n in huecos %}
    <span class="mo-hueco" data-tag="{{ tag|e }}">{{ tag }} <b>{{ n }}</b></span>
    {% endfor %}
  </div>
</div>
{% endif %}

<details class="mo-sec" style="border:1px solid var(--acc2); border-radius:10px; padding:.6rem .9rem">
  <summary>🤖 Generar borradores desde mis fotos <span class="hint-ico" title="Recorre tu biblioteca, junta las combinaciones de etiquetas que se repiten (una situación real, no un trío casual) y le pide al modelo 3 pies borrador de cada una — sanos o crudos según la carpeta. Después los corregís acá: el modelo escribe medio mal, esto te da el primer borrador, no el pie final. Lento: es de a una, dejalo corriendo.">ⓘ</span></summary>
  <div class="row" style="margin:.4rem 0">
    <label>Solo combinaciones que aparezcan en al menos</label>
    <input type="number" id="gen-k" value="3" min="1" max="50" style="width:4rem">
    <label>fotos</label>
    <button class="primary" id="gen-go" onclick="genGo()">▶ Generar</button>
    <button class="secondary" id="gen-stop" onclick="genStop()" style="display:none">⏹ Frenar</button>
  </div>
  <div id="gen-prog" style="display:none">
    <progress id="gen-bar" style="width:100%"></progress>
    <p class="meta"><b id="gen-cur">0</b>/<b id="gen-tot">0</b> · <span id="gen-combo"></span>
       · <b id="gen-frases">0</b> frases nuevas</p>
    <div id="gen-ultimos" class="meta" style="font-size:.82rem"></div>
  </div>
</details>

<div class="panel">
  <div class="row">
    <button class="primary" onclick="cmbAbrir()" title="Elegís etiquetas y sacás/editás el JSON de esa combinación">🔍 Armar / editar una combinación</button>
    <button class="secondary" onclick="mog('mo-file').click()" title="Cargá un .json y FUSIONA con lo que tenés">⬆ Importar archivo</button>
    <input type="file" id="mo-file" accept=".json,application/json" style="display:none" onchange="moImportar(this)">
    <label class="mo-nsfw" style="flex-direction:row"><input type="checkbox" id="imp-reemp"> reemplazar todo</label>
    <a class="meta" href="{{ url_for('moldes_export') }}" style="text-decoration:underline">backup completo</a>
  </div>
</div>

{# la lista completa de moldes va COLAPSADA: molestaba tenerla siempre a la vista.
   El trabajo normal es por el modal de arriba; esto es para revisar/editar en bloque. #}
<details class="mo-sec" style="border:1px solid var(--bd); border-radius:10px; padding:.6rem .9rem">
  <summary>📋 Ver / editar todos los moldes ({{ cuerpos|length }})</summary>
  <div class="row" style="margin:.5rem 0">
    <input id="mo-filtro" class="path" placeholder="🔍 filtrar por texto o etiqueta…" oninput="moFiltrar()">
    <span class="meta"><b id="mo-count">{{ cuerpos|length }}</b> molde(s)</span>
  </div>
  <div class="mo-head">
    <span>se dispara con (etiquetas, coma)</span><span>pies — uno por línea, elige al azar</span><span></span><span></span>
  </div>
  <div id="mo-lista">
    {% for tags, textos, nsfw, borrador in cuerpos %}
    <div class="mo-cuerpo{% if borrador %} es-borrador{% endif %}">
      <input class="tags" value="{{ tags|join(', ') }}" placeholder="playa, sol" spellcheck="false">
      <textarea class="textos" rows="{{ [textos|length, 1]|max }}" placeholder="un pie por línea, en primera persona" spellcheck="false">{{ textos|join('\n') }}</textarea>
      <label class="mo-nsfw"><input type="checkbox" class="nsfw" {% if nsfw %}checked{% endif %}> 🔞{% if borrador %}<br><span class="mo-bdg" title="Lo generó el modelo, revisalo. Al guardar deja de ser borrador.">borrador</span>{% endif %}</label>
      <button class="mo-x" title="Borrar" onclick="this.closest('.mo-cuerpo').remove(); moContar()">✕</button>
    </div>
    {% endfor %}
  </div>
  <div class="row" style="margin-top:.7rem">
    <button class="secondary" onclick="moAgregar()">＋ molde nuevo</button>
    <button class="primary" onclick="moGuardar()">💾 Guardar</button>
    <button class="secondary" onclick="moFabrica()" title="Vuelve a los moldes que trae el programa">↺ restaurar los de fábrica</button>
    <span class="meta" id="mo-st"></span>
  </div>
</details>

<div id="cmb-modal" class="modal-ov" style="display:none" onclick="if(event.target===this) cmbCerrar()">
  <div class="modal-box" style="width:600px">
    <div class="row" style="justify-content:space-between;margin:0 0 .5rem">
      <b style="font-size:1.05rem">🔍 Combinación de etiquetas <span class="hint-ico" title="Buscá etiquetas que estén en tus fotos y agregá las que quieras (para un pie específico de una foto, metelas todas). Podés pegar varias con coma y Enter. &quot;Armar&quot; te da el JSON de esa combinación para copiar, editar y guardar.">ⓘ</span></b>
      <button class="secondary sm" onclick="cmbCerrar()" title="Cerrar">✕</button>
    </div>
    <input id="cmb-buscar" class="path" placeholder="🔍 buscar o pegar etiquetas (coma) y Enter…" autocomplete="off"
           oninput="cmbSug()" onfocus="cmbSug()" onkeydown="cmbKey(event)" style="width:100%">
    <div id="cmb-sug" style="max-height:9rem;overflow-y:auto;margin:.3rem 0"></div>
    <div id="cmb-chips" style="margin:.3rem 0;min-height:1.6rem"></div>
    <div class="row" style="margin:.3rem 0">
      <label class="mo-nsfw" style="flex-direction:row"><input type="checkbox" id="cmb-nsfw"> 🔞 versión cruda</label>
      <button class="primary" onclick="cmbArmar()">Armar JSON ▾</button>
    </div>
    <textarea id="cmb-json" class="textos" spellcheck="false" rows="8"
              placeholder="Acá aparece el JSON de la combinación al Armar. Editalo y Guardá." style="width:100%"></textarea>
    <div class="row" style="margin-top:.4rem">
      <button class="secondary" onclick="cmbCopiar()">📋 Copiar</button>
      <button class="primary" onclick="cmbGuardar()">💾 Guardar (fusiona)</button>
      <span class="meta" id="cmb-st"></span>
    </div>
  </div>
</div>

<details class="mo-sec">
  <summary>Cierres — la coletilla corta que a veces se le suma al pie <span class="hint-ico" title="Se pegan a cualquier molde, en 2 de cada 3 pies («Sola en casa. No me juzguen.»). Los 🔞 solo con moldes 🔞.">ⓘ</span></summary>
  <div class="row" style="margin:.3rem 0 0">
    <button class="secondary" onclick="location.href='{{ url_for('cierres_export') }}'"
            title="Baja los cierres como .json">⬇ Exportar cierres</button>
    <button class="secondary" onclick="mog('ci-file').click()"
            title="Cargá un .json de cierres y los suma a los que tenés">⬆ Importar cierres</button>
    <input type="file" id="ci-file" accept=".json,application/json" style="display:none"
           onchange="ciImportar(this)">
    <span class="meta" id="ci-st"></span>
  </div>
  <div id="mo-cierres">
    {% for texto, nsfw in cierres %}
    <div class="mo-cuerpo" style="grid-template-columns:minmax(0,1fr) auto 1.6rem">
      <input class="texto" value="{{ texto|e }}" placeholder="una coletilla corta" spellcheck="false">
      <label class="mo-nsfw"><input type="checkbox" class="nsfw" {% if nsfw %}checked{% endif %}> 🔞</label>
      <button class="mo-x" title="Borrar" onclick="this.closest('.mo-cuerpo').remove()">✕</button>
    </div>
    {% endfor %}
  </div>
  <button class="secondary" style="margin-top:.5rem" onclick="moAgregarCierre()">＋ cierre</button>
</details>

<details class="mo-sec">
  <summary>Alias — sinónimos y tipeos que cuentan como una etiqueta conocida <span class="hint-ico" title="Tus etiquetas vienen con variantes (denuda, en bolas). Acá decís que todas ésas valen como una sola, así el molde igual se prende (denuda → desnuda).">ⓘ</span></summary>
  <div id="mo-alias">
    {% for k, v in alias %}
    <div class="mo-cuerpo" style="grid-template-columns:minmax(0,1fr) 1.2rem minmax(0,1fr) 1.6rem">
      <input class="de" value="{{ k|e }}" placeholder="lo que aparece" spellcheck="false">
      <span class="cv-arrow" style="text-align:center;color:var(--mut)">→</span>
      <input class="a" value="{{ v|e }}" placeholder="cuenta como" spellcheck="false">
      <button class="mo-x" title="Borrar" onclick="this.closest('.mo-cuerpo').remove()">✕</button>
    </div>
    {% endfor %}
  </div>
  <button class="secondary" style="margin-top:.5rem" onclick="moAgregarAlias()">＋ alias</button>
</details>

<script>
const mog = id => document.getElementById(id);
function moContar() {
  mog('mo-count').textContent = document.querySelectorAll('#mo-lista .mo-cuerpo').length;
}
function moFiltrar() {
  const q = mog('mo-filtro').value.trim().toLowerCase();
  document.querySelectorAll('#mo-lista .mo-cuerpo').forEach(row => {
    const a = row.querySelector('.tags').value.toLowerCase();
    const b = row.querySelector('.textos').value.toLowerCase();
    row.style.display = (!q || a.includes(q) || b.includes(q)) ? '' : 'none';
  });
}
function moFilaCuerpo(tags, textos, nsfw) {
  const row = document.createElement('div'); row.className = 'mo-cuerpo';
  const t = document.createElement('input'); t.className = 'tags'; t.placeholder = 'playa, sol';
  t.spellcheck = false; t.value = tags || '';
  const x = document.createElement('textarea'); x.className = 'textos';
  x.placeholder = 'un pie por línea, en primera persona'; x.spellcheck = false;
  x.rows = 2; x.value = textos || '';
  const lab = document.createElement('label'); lab.className = 'mo-nsfw';
  const cb = document.createElement('input'); cb.type = 'checkbox'; cb.className = 'nsfw'; cb.checked = !!nsfw;
  lab.appendChild(cb); lab.appendChild(document.createTextNode(' 🔞'));
  const del = document.createElement('button'); del.className = 'mo-x'; del.textContent = '✕'; del.title = 'Borrar';
  del.onclick = () => { row.remove(); moContar(); };
  row.append(t, x, lab, del);
  return row;
}
function moAgregar(tags, textos, nsfw) {
  const row = moFilaCuerpo(tags, textos, nsfw);
  mog('mo-lista').appendChild(row); moContar();
  return row;
}
// Los chips de hueco/combinación llevan la etiqueta en un data-, no en el
// onclick: una etiqueta tuya con apóstrofo ("dia d'campo") partía el string de
// JS y rompía el atributo. Por delegación, así también anda lo que se agregue
// después.
document.addEventListener('click', e => {
  const chip = e.target.closest('.mo-hueco[data-tag]');
  if (chip) moDesdeHueco(chip.dataset.tag);
});
function moDesdeCombo(el) {    // clic en una combinación sin cubrir: la abre en el modal
  cmbAbrir();
  (el.dataset.combo || '').split(',').map(s => s.trim()).filter(Boolean).forEach(cmbAdd);
  cmbArmar();
}
function moDesdeHueco(tag) {   // clic en una etiqueta sin molde: arranca un molde con ella
  const row = moAgregar(tag, '', false);
  row.scrollIntoView({behavior: 'smooth', block: 'center'});
  row.querySelector('.textos').focus();
}
function moAgregarCierre() {
  const row = document.createElement('div'); row.className = 'mo-cuerpo';
  row.style.gridTemplateColumns = 'minmax(0,1fr) auto 1.6rem';
  const x = document.createElement('input'); x.className = 'texto'; x.placeholder = 'una coletilla corta'; x.spellcheck = false;
  const lab = document.createElement('label'); lab.className = 'mo-nsfw';
  const cb = document.createElement('input'); cb.type = 'checkbox'; cb.className = 'nsfw';
  lab.appendChild(cb); lab.appendChild(document.createTextNode(' 🔞'));
  const del = document.createElement('button'); del.className = 'mo-x'; del.textContent = '✕';
  del.onclick = () => row.remove();
  row.append(x, lab, del); mog('mo-cierres').appendChild(row); x.focus();
}
function moAgregarAlias() {
  const row = document.createElement('div'); row.className = 'mo-cuerpo';
  row.style.gridTemplateColumns = 'minmax(0,1fr) 1.2rem minmax(0,1fr) 1.6rem';
  const de = document.createElement('input'); de.className = 'de'; de.placeholder = 'lo que aparece'; de.spellcheck = false;
  const fl = document.createElement('span'); fl.textContent = '→'; fl.style.textAlign = 'center'; fl.style.color = 'var(--mut)';
  const a = document.createElement('input'); a.className = 'a'; a.placeholder = 'cuenta como'; a.spellcheck = false;
  const del = document.createElement('button'); del.className = 'mo-x'; del.textContent = '✕';
  del.onclick = () => row.remove();
  row.append(de, fl, a, del); mog('mo-alias').appendChild(row); de.focus();
}
function moGuardar() {
  const cuerpos = [];
  document.querySelectorAll('#mo-lista .mo-cuerpo').forEach(row => {
    const tags = row.querySelector('.tags').value.split(',').map(s => s.trim()).filter(Boolean);
    const textos = row.querySelector('.textos').value.split('\\n').map(s => s.trim()).filter(Boolean);
    if (tags.length && textos.length) cuerpos.push({tags: tags, textos: textos, nsfw: row.querySelector('.nsfw').checked});
  });
  const cierres = [];
  document.querySelectorAll('#mo-cierres .mo-cuerpo').forEach(row => {
    const texto = row.querySelector('.texto').value.trim();
    if (texto) cierres.push({texto: texto, nsfw: row.querySelector('.nsfw').checked});
  });
  const alias = {};
  document.querySelectorAll('#mo-alias .mo-cuerpo').forEach(row => {
    const de = row.querySelector('.de').value.trim(), a = row.querySelector('.a').value.trim();
    if (de && a) alias[de] = a;
  });
  mog('mo-st').textContent = ' guardando…';
  fetch("{{ url_for('moldes_save') }}", {method: 'POST',
    headers: {'Content-Type': 'application/x-www-form-urlencoded'},
    body: new URLSearchParams({cuerpos: JSON.stringify(cuerpos), cierres: JSON.stringify(cierres),
      alias: JSON.stringify(alias)})})
    .then(r => r.json()).then(d => {
      mog('mo-st').textContent = d.ok ? (' ✓ ' + d.cuerpos + ' moldes guardados. Recargá para ver los huecos al día.') : ' error';
      setTimeout(() => mog('mo-st').textContent = '', 6000);
    }).catch(e => mog('mo-st').textContent = ' ' + e);
}
function moFabrica() {
  if (!confirm('¿Volver a los moldes de fábrica? Se pierden los tuyos.')) return;
  fetch("{{ url_for('moldes_save') }}", {method: 'POST',
    headers: {'Content-Type': 'application/x-www-form-urlencoded'},
    body: new URLSearchParams({fabrica: '1'})})
    .then(r => r.json()).then(() => location.reload());
}
// --- modal de combinación: elegís etiquetas vivas y sacás/editás su JSON ---
function norm(s) {   // minúsculas y sin tildes, para comparar sin exactitud
  return (s || '').toLowerCase().normalize('NFD').replace(/[\\u0300-\\u036f]/g, '');
}
const CMB_VIVAS = {{ vivas|tojson }};   // [[etiqueta, en cuantas fotos], ...]
let cmbSel = [];                        // etiquetas elegidas (máx 3)
function cmbAbrir() {
  cmbSel = []; cmbRender(); mog('cmb-json').value = ''; mog('cmb-st').textContent = '';
  mog('cmb-buscar').value = ''; mog('cmb-sug').innerHTML = '';
  mog('cmb-modal').style.display = 'flex'; mog('cmb-buscar').focus();
}
function cmbCerrar() { mog('cmb-modal').style.display = 'none'; }
function cmbSug() {
  const q = norm(mog('cmb-buscar').value.trim());
  const box = mog('cmb-sug'); box.innerHTML = '';
  // sin texto: las 10 más usadas (CMB_VIVAS ya viene ordenada por uso). Con
  // texto: las que coinciden.
  let hits;
  if (!q) {
    hits = CMB_VIVAS.filter(([t]) => !cmbSel.includes(t)).slice(0, 10);
    if (hits.length) {
      const h = document.createElement('div'); h.className = 'meta'; h.style.padding = '.2rem .3rem';
      h.textContent = 'más usadas:'; box.appendChild(h);
    }
  } else {
    hits = CMB_VIVAS.filter(([t]) => norm(t).includes(q) && !cmbSel.includes(t)).slice(0, 12);
  }
  if (!hits.length) {
    const d = document.createElement('div'); d.className = 'meta'; d.style.padding = '.3rem';
    d.textContent = q ? 'sin etiquetas vivas que coincidan' : 'no hay etiquetas en tus fotos todavía';
    box.appendChild(d); return;
  }
  hits.forEach(([t, n]) => {
    const d = document.createElement('div'); d.className = 'pmodal-item';
    d.textContent = t + ' · ' + n + ' fotos';
    d.onclick = () => cmbAdd(t);            // listener, no string: la etiqueta va literal
    box.appendChild(d);
  });
}
function cmbAdd(t) {
  t = (t || '').trim();
  if (t && !cmbSel.includes(t)) cmbSel.push(t);   // sin tope: una foto puede tener muchas
  mog('cmb-buscar').value = ''; mog('cmb-sug').innerHTML = ''; cmbRender();
  mog('cmb-buscar').focus();
}
function cmbDel(t) { cmbSel = cmbSel.filter(x => x !== t); cmbRender(); }
function cmbRender() {
  const box = mog('cmb-chips'); box.innerHTML = '';
  if (!cmbSel.length) {
    const s = document.createElement('span'); s.className = 'meta';
    s.textContent = 'todavía no agregaste ninguna'; box.appendChild(s); return;
  }
  cmbSel.forEach(t => {
    const chip = document.createElement('span'); chip.className = 'chip';
    chip.textContent = t + ' ';
    const x = document.createElement('b'); x.textContent = '✕'; x.style.cursor = 'pointer';
    x.onclick = () => cmbDel(t);
    chip.appendChild(x); box.appendChild(chip);
  });
}
function cmbKey(e) {   // Enter: agrega (soporta pegar varias con coma)
  if (e.key !== 'Enter') return;
  e.preventDefault();
  const raw = mog('cmb-buscar').value.trim();
  if (!raw) return;
  const partes = raw.split(',').map(s => s.trim()).filter(Boolean);
  if (partes.length > 1) { partes.forEach(cmbAdd); return; }   // lista pegada
  // una sola: si coincide con una viva, esa; si no, lo tipeado tal cual (para
  // etiquetas que quizás no estén en el buscador pero sí en la foto)
  const q = norm(raw);
  const hit = CMB_VIVAS.find(([t]) => norm(t).includes(q) && !cmbSel.includes(t));
  cmbAdd(hit ? hit[0] : raw);
}
function cmbArmar() {
  if (!cmbSel.length) { alert('Agregá al menos una etiqueta.'); return; }
  const nsfw = mog('cmb-nsfw').checked;
  const claveSel = cmbSel.map(norm).sort().join('|');
  // buscar en el editor un molde con EXACTAMENTE estas etiquetas y este registro
  let encontrado = null;
  document.querySelectorAll('#mo-lista .mo-cuerpo').forEach(row => {
    const tags = row.querySelector('.tags').value.split(',').map(s => s.trim()).filter(Boolean);
    const clave = tags.map(norm).sort().join('|');
    if (clave === claveSel && row.querySelector('.nsfw').checked === nsfw) {
      encontrado = row.querySelector('.textos').value.split('\\n').map(s => s.trim()).filter(Boolean);
    }
  });
  const cuerpo = {tags: cmbSel.slice(), textos: encontrado || ['', '', ''], nsfw: nsfw};
  mog('cmb-json').value = JSON.stringify({cuerpos: [cuerpo]}, null, 2);
  mog('cmb-st').textContent = encontrado ? ' (ya existía, lo traje)' : ' (nuevo: escribí los pies)';
}
function cmbCopiar() {
  const t = mog('cmb-json'); t.select();
  navigator.clipboard.writeText(t.value).then(
    () => mog('cmb-st').textContent = ' ✓ copiado',
    () => { document.execCommand('copy'); mog('cmb-st').textContent = ' ✓ copiado'; });
}
function cmbGuardar() {
  const data = mog('cmb-json').value;
  if (!data.trim()) { mog('cmb-st').textContent = ' escribí algo primero'; return; }
  const fd = new FormData(); fd.append('texto', data);   // fusiona; server tolerante
  mog('cmb-st').textContent = ' guardando…';
  fetch("{{ url_for('moldes_import') }}", {method: 'POST', body: fd})
    .then(r => r.json()).then(d => {
      if (d.ok) { mog('cmb-st').textContent = ' ✓ guardado, recargando…'; setTimeout(() => location.reload(), 900); }
      else mog('cmb-st').textContent = ' ✗ ' + d.error;
    }).catch(e => mog('cmb-st').textContent = ' ' + e);
}
document.addEventListener('keydown', e => {   // Escape cierra el modal
  if (e.key === 'Escape' && mog('cmb-modal').style.display !== 'none') cmbCerrar();
});

let genTimer = null;
function genGo() {
  const k = mog('gen-k').value || 3;
  fetch("{{ url_for('moldes_generar') }}", {method: 'POST',
    headers: {'Content-Type': 'application/x-www-form-urlencoded'},
    body: new URLSearchParams({k: k})})
    .then(r => r.json()).then(d => {
      if (!d.ok) { alert(d.error || 'no se pudo'); return; }
      mog('gen-go').style.display = 'none';
      mog('gen-stop').style.display = '';
      mog('gen-prog').style.display = '';
      genTimer = setInterval(genPoll, 1500);
    });
}
function genStop() {
  fetch("{{ url_for('moldes_generar_stop') }}", {method: 'POST'});
}
function genPoll() {
  fetch("{{ url_for('moldes_generar_estado') }}").then(r => r.json()).then(s => {
    mog('gen-cur').textContent = s.current; mog('gen-tot').textContent = s.total;
    mog('gen-combo').textContent = s.combo || ''; mog('gen-frases').textContent = s.frases;
    const bar = mog('gen-bar'); bar.max = s.total || 1; bar.value = s.current;
    mog('gen-ultimos').innerHTML = (s.ultimos || []).map(u =>
      '· ' + u.replace(/[<>&]/g, c => ({'<':'&lt;','>':'&gt;','&':'&amp;'}[c]))).join('<br>');
    if (s.status !== 'running') {
      clearInterval(genTimer); genTimer = null;
      mog('gen-go').style.display = ''; mog('gen-stop').style.display = 'none';
      if (s.status === 'error') { mog('gen-combo').textContent = '✗ ' + s.error; return; }
      mog('gen-combo').textContent = '✓ listo: ' + s.combos_nuevos + ' combos, ' + s.frases + ' frases. Recargando…';
      setTimeout(() => location.reload(), 1500);
    }
  });
}
function ciImportar(input) {   // importar SOLO cierres (fusiona con los que hay)
  const f = input.files[0]; if (!f) return;
  const fd = new FormData(); fd.append('archivo', f);
  mog('ci-st').textContent = ' importando…';
  fetch("{{ url_for('moldes_import') }}", {method: 'POST', body: fd})
    .then(r => r.json()).then(d => {
      input.value = '';
      if (!d.ok) { mog('ci-st').textContent = ' ✗ ' + d.error; return; }
      mog('ci-st').textContent = ' ✓ importados, recargando…';
      setTimeout(() => location.reload(), 800);
    }).catch(e => { mog('ci-st').textContent = ' ' + e; input.value = ''; });
}

function moImportar(input) {
  const f = input.files[0]; if (!f) return;
  const reemp = mog('imp-reemp').checked;
  const msg = reemp ? 'Reemplazar TODOS tus moldes por los del archivo. ¿Seguir?'
                    : 'Fusionar el archivo con tus moldes (no borra el resto). ¿Seguir?';
  if (!confirm(msg)) { input.value = ''; return; }
  const fd = new FormData(); fd.append('archivo', f);
  if (reemp) fd.append('reemplazar', '1');
  mog('mo-st').textContent = ' importando…';
  fetch("{{ url_for('moldes_import') }}", {method: 'POST', body: fd})
    .then(r => r.json()).then(d => {
      if (d.ok) { mog('mo-st').textContent = ' ✓ ' + d.cuerpos + ' moldes importados'; location.reload(); }
      else { mog('mo-st').textContent = ' ✗ ' + d.error; }
      input.value = '';
    }).catch(e => { mog('mo-st').textContent = ' ' + e; input.value = ''; });
}
</script>
"""


@app.route("/moldes")
def moldes_page():
    """Editor de moldes de pie: cuerpos, cierres, alias, y los huecos de cobertura."""
    conteo = tags_mod.all_photo_tag_counts()
    # etiquetas "vivas": las que de verdad están en fotos, con en cuántas. Para el
    # buscador del modal de combinaciones (así se ve qué se usa y qué no).
    vivas = sorted(conteo.items(), key=lambda kv: (-kv[1], kv[0]))
    # cobertura: por cada foto etiquetada, ¿le prende algún molde o cae al modelo?
    fotos_cob = []
    for clave, _ in tags_mod.all_photo_entries():
        if "/" not in clave:
            continue
        persona, rel = clave.split("/", 1)
        fotos_cob.append((tags_mod.effective_photo(persona, rel),
                          posts_mod.identidad_foto(persona, rel)["sano"]))
    cob = moldes_mod.cobertura_de(fotos_cob) if fotos_cob else None
    return render_template_string(
        shell("moldes", MOLDES_TEMPLATE),
        cuerpos=moldes_mod.combinaciones(), cierres=moldes_mod.cierres(),
        alias=sorted(moldes_mod.alias().items(), key=lambda kv: kv[0]),
        vivas=vivas, cob=cob,
        huecos=moldes_mod.huecos(conteo)[:40],
        message=request.args.get("msg", ""))


@app.route("/moldes/save", methods=["POST"])
def moldes_save():
    """Reemplaza los moldes con lo que quedó en la web. `fabrica=1` borra los
    tuyos y vuelve a los del código."""
    if request.form.get("fabrica") == "1":
        moldes_mod.MOLDES_PATH.unlink(missing_ok=True)
        return {"ok": True, "cuerpos": len(moldes_mod.CUERPOS_FABRICA)}
    try:
        crudos = json.loads(request.form.get("cuerpos", "[]"))
        cierres_j = json.loads(request.form.get("cierres", "[]"))
        alias_j = json.loads(request.form.get("alias", "{}"))
    except Exception:
        return {"ok": False, "error": "datos ilegibles"}
    # forma agrupada: cada molde trae un array de variantes (textos)
    cuerpos = []
    for c in crudos:
        tags = c.get("tags") or []
        textos = [t.strip() for t in (c.get("textos") or []) if str(t).strip()]
        if tags and textos:
            cuerpos.append((tags, textos, bool(c.get("nsfw"))))
    cierres = [(c["texto"].strip(), bool(c.get("nsfw")))
               for c in cierres_j if c.get("texto", "").strip()]
    alias = {str(k).strip(): str(v).strip() for k, v in alias_j.items()
             if str(k).strip() and str(v).strip()}
    moldes_mod.guardar(cuerpos_=cuerpos, cierres_=cierres, alias_=alias)
    return {"ok": True, "cuerpos": len(cuerpos)}


def _gen_worker(k, max_tags):
    """Recolecta combos de toda la biblioteca (con umbral k) y le pide al modelo
    3 frases borrador por cada uno, de a una. Corre en su propio hilo."""
    try:
        entradas = tags_mod.all_photo_entries()
        # (tags, es_nsfw) por foto: la carpeta _nsfw decide el registro
        fotos = [(t, NSFW_DIR_NAME in Path(k2.split("/", 1)[1] if "/" in k2 else k2).parts)
                 for k2, t in entradas]
        combos = moldes_mod.recolectar_combos(fotos, k=k, max_tags=max_tags)
        GEN_STATE.update(total=len(combos), current=0, combos_nuevos=0, frases=0)
        for i, (ctags, nsfw) in enumerate(combos, 1):
            if GEN_STATE["stop"]:
                break
            GEN_STATE.update(current=i, combo=("🔞 " if nsfw else "") + ", ".join(ctags))
            frases = moldes_mod.generar_frases(ctags, nsfw, n=3)
            if frases:
                cn, fn = moldes_mod.fusionar_borradores([(ctags, frases, nsfw)])
                GEN_STATE["combos_nuevos"] += cn
                GEN_STATE["frases"] += fn
                GEN_STATE["ultimos"] = ([f"{', '.join(ctags)}: {frases[0]}"]
                                        + GEN_STATE["ultimos"])[:6]
        GEN_STATE["status"] = "stopped" if GEN_STATE["stop"] else "done"
    except Exception as e:
        GEN_STATE.update(status="error", error=str(e))


@app.route("/moldes/generar", methods=["POST"])
def moldes_generar():
    """Arranca el generador de borradores en segundo plano."""
    if GEN_STATE["status"] == "running":
        return {"ok": False, "error": "Ya está corriendo."}
    try:
        k = max(1, min(int(request.form.get("k", 3)), 50))
    except ValueError:
        k = 3
    GEN_STATE.update(status="running", stop=False, current=0, total=0, combo="",
                     error="", combos_nuevos=0, frases=0, empezo=time.time(), ultimos=[])
    threading.Thread(target=_gen_worker, args=(k, 3), daemon=True).start()
    return {"ok": True}


@app.route("/moldes/generar/estado")
def moldes_generar_estado():
    return dict(GEN_STATE)


@app.route("/moldes/generar/stop", methods=["POST"])
def moldes_generar_stop():
    GEN_STATE["stop"] = True
    return {"ok": True}


@app.route("/moldes/export")
def moldes_export():
    """Baja los moldes actuales como .json (para backup o pasarlos a otra PC)."""
    data = json.dumps(moldes_mod.exportar(), ensure_ascii=False, indent=2)
    return Response(data, mimetype="application/json",
                    headers={"Content-Disposition":
                             "attachment; filename=pies-moldes.json"})


@app.route("/cierres/export")
def cierres_export():
    """Baja SOLO los cierres como .json (emojis tal cual, sin escapar).

    Sale envuelto en {"cierres": [...]} — la misma forma que espera el importador,
    así el botón "Importar archivo" los trae de vuelta. Antes bajaba un array
    pelado que el importador rechazaba: exportar no servía para nada."""
    data = json.dumps({"cierres": [{"texto": x, "nsfw": n} for x, n in moldes_mod.cierres()]},
                      ensure_ascii=False, indent=2)
    return Response(data, mimetype="application/json",
                    headers={"Content-Disposition":
                             "attachment; filename=cierres.json"})


@app.route("/moldes/import", methods=["POST"])
def moldes_import():
    """Reemplaza los moldes con un .json subido (o pegado en el textarea)."""
    crudo = ""
    f = request.files.get("archivo")
    if f:
        crudo = f.read().decode("utf-8", "replace")
    else:
        crudo = request.form.get("texto", "")
    try:
        # tolerante: saca cercas ```json, comillas curvas, comas colgando y prosa
        # alrededor. Los emojis pasan tal cual. Ver moldes.json_laxo.
        data = json.loads(moldes_mod.json_laxo(crudo))
    except Exception:
        return {"ok": False, "error": "No pude leer el JSON (revisá que las llaves { } y las comillas estén completas)."}
    try:
        n = moldes_mod.importar(data, reemplazar=request.form.get("reemplazar") == "1")
    except ValueError as e:
        return {"ok": False, "error": str(e)}
    return {"ok": True, "cuerpos": n}


@app.route("/moldes/combo", methods=["POST"])
def moldes_combo():
    """El JSON de UNA combinación (para el botón "armar pie" del visor): el molde
    que exista para esas etiquetas, o un esqueleto vacío."""
    tags = [t.strip() for t in request.form.get("tags", "").split(",") if t.strip()]
    perfil = [t.strip() for t in request.form.get("perfil", "").split(",") if t.strip()]
    nsfw = request.form.get("nsfw") == "1"
    data = moldes_mod.json_combo(tags, nsfw, tags_perfil=perfil)
    return {"ok": True, "json": json.dumps(data, ensure_ascii=False, indent=2)}


@app.route("/moldes/desde_foto/<name>", methods=["POST"])
def moldes_desde_foto(name):
    """Guarda el molde del ✍ del visor Y suma sus etiquetas a la foto.

    IMPORTANTE — SOLO AGREGA, nunca pisa ni borra: las etiquetas del JSON se
    UNEN a las que la foto ya tiene (después de la pérdida de datos de hoy, esto
    no se negocia). Para SACAR una etiqueta de una foto está el gestor 🏷, no
    esto."""
    rel = request.form.get("rel", "")
    try:
        data = json.loads(moldes_mod.json_laxo(request.form.get("texto", "")))
    except Exception:
        return {"ok": False, "error": "No pude leer el JSON (revisá llaves y comillas)."}
    try:
        n = moldes_mod.importar(data)          # fusiona el molde (no reemplaza)
    except ValueError as e:
        return {"ok": False, "error": str(e)}
    # etiquetas del JSON -> a la foto, en UNIÓN con las que ya tiene.
    # SOLO `tags` (las de la foto): `tags_perfil` (relación: hermana, mujer) se
    # ignora acá a propósito — ya vive en el perfil y la hereda toda la persona,
    # meterla en cada foto sería duplicarla.
    del_json = [str(t).strip() for c in (data.get("cuerpos") or [])
                for t in (c.get("tags") or []) if str(t).strip()]
    tags_foto = tags_mod.get_photo(name, rel)
    if del_json:
        # set_photo normaliza y saca duplicados; le paso EXISTENTES + nuevas, así
        # la unión conserva todo lo previo y solo agrega lo que falta
        tags_foto = tags_mod.set_photo(name, rel, tags_foto + del_json)
    return {"ok": True, "cuerpos": n, "tags": tags_foto}


@app.route("/tags/merge", methods=["POST"])
def tags_merge():
    """Junta dos etiquetas casi iguales en TODA la base (denuda -> desnuda)."""
    vieja = request.form.get("vieja", "")
    nueva = request.form.get("nueva", "")
    if not vieja or not nueva:
        return {"ok": False, "error": "Faltan las etiquetas."}
    return {"ok": True, "n": tags_mod.merge_tag_global(vieja, nueva)}


@app.route("/conversiones")
def conversiones_page():
    """Gestor del diccionario de conversiones (rusa->paja rusa): ver, editar, borrar."""
    cfg = load_config()
    conv = cfg.get("f_conversiones", {}) or {}
    pares = sorted(conv.items(), key=lambda kv: kv[0].lower())
    return render_template_string(
        shell("conversiones", CONVERSIONES_TEMPLATE),
        pares=pares, parecidas=tags_mod.pares_parecidos()[:30],
        message=request.args.get("msg", ""))


@app.route("/conversiones/save", methods=["POST"])
def conversiones_save():
    """Reemplaza TODO el diccionario con lo que quedó en la lista (editado/borrado)."""
    try:
        pares = json.loads(request.form.get("pares", "[]"))
    except Exception:
        pares = []
    nuevo = {}
    for p in pares:
        k = str(p.get("orig", "")).strip()
        v = str(p.get("final", "")).strip()
        if k and v and k.lower() != v.lower():
            nuevo[k] = v                          # si repetís una clave, gana la última
    cfg = load_config()
    cfg["f_conversiones"] = nuevo
    save_config(cfg)
    return {"ok": True, "n": len(nuevo)}


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
      data-confirm="Se reemplazan TODOS los ejemplos de {{ name }} por las caras verificadas. ¿Aplicar?">
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
<h1>Revisar NSFW <span class="hint-ico" title="Revisa una carpeta buscando fotos con desnudez explícita (todo local, no sale nada de esta máquina). Solo marca las clases inequívocas — una foto normal de playa no debería saltar. Las sospechosas se muestran para que las revises: las que confirmes se mueven a una subcarpeta {{ nsfw_dir }} dentro de la carpeta revisada. No se borra nada.">ⓘ</span></h1>

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
      <input type="range" class="sens" name="sens" min="0" max="100" value="{{ sens }}"
             style="--pct:{{ sens }}%" oninput="this.style.setProperty('--pct', this.value + '%')">
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
  <div class="counter"><span id="nsel" style="color:#f0883e">0</span> seleccionada(s)
    <span class="meta">· {{ state.results|length }} sospechosa(s) en {{ state.scanned }} foto(s) revisadas</span></div>
  {% if state.results %}
  <form method="post" action="{{ url_for('nsfw_move') }}"
        data-confirm="Se moverán las fotos tildadas a {{ nsfw_dir }} dentro de la carpeta revisada. ¿Continuar?">
    <div class="row">
      <button class="organize">📦 Mover las seleccionadas a {{ nsfw_dir }}</button>
      <button type="button" class="secondary" onclick="toggleAll(this)">☑ Seleccionar / deseleccionar todas</button>
      <span class="hint-ico" title="Destildá los falsos positivos (playa, etc.) antes de mover · clic en la imagen abre el visor (zoom, arrastre, ← →).">ⓘ</span>
    </div>
    <div class="nsfw-grid">
      {% for r in state.results %}
      <div class="hit">
        <div class="hit-thumb">
          {% if r.thumb %}
          <img src="{{ url_for('nsfw_thumb', fn=r.thumb) }}" alt="{{ r.rel }}"
               onclick="lbOpen({{ loop.index0 }})">
          {% endif %}
          <label class="hit-pill" title="{{ r.rel.split('/')[-1] }}">
            <input type="checkbox" name="rel" value="{{ r.rel }}" checked onchange="nselCount()">
            {{ (r.score * 100)|int }}%
          </label>
          <span class="hit-tags" title="{{ r.rel }}">{{ r.labels|join(', ') }}</span>
        </div>
      </div>
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
  const boxes = [...btn.closest('form').querySelectorAll('.nsfw-grid input[type=checkbox]')];
  const all = boxes.every(b => b.checked);
  boxes.forEach(b => b.checked = !all);
  nselCount();
}
function nselCount() {
  const el = document.getElementById('nsel');
  if (el) el.textContent = document.querySelectorAll('.nsfw-grid input:checked').length;
}
nselCount();
</script>
"""

# el visor compartido también en la revisión NSFW
NSFW_TEMPLATE = NSFW_TEMPLATE.replace("__LIGHTBOX__", LIGHTBOX)


# --- playlist de favoritos de una persona (mockup Favoritos) -----------------
FAVS_TEMPLATE = """
<a href="{{ url_for('person_page', name=name) }}">← Volver a {{ name }}</a>

<div class="pcard" style="grid-template-columns:120px 1fr; margin-top:.8rem">
  <div class="pcard-cover" style="padding:14px">
    <div class="cover-box">
      {% if has_thumb %}<img src="{{ url_for('known_thumb', name=name) }}?v={{ cache_bust }}" alt="{{ name }}">
      {% else %}<div class="noimg">👤</div>{% endif %}
    </div>
  </div>
  <div class="pcard-main" style="justify-content:center">
    <div class="meta" style="text-transform:uppercase; letter-spacing:.06em; font-weight:600">⭐ Favoritas</div>
    <h1 style="margin:0; font-size:24px">{{ name }}</h1>
    <div class="meta"><b style="color:var(--tx)" id="favcount">{{ favs|length }}</b> fotos guardadas</div>
  </div>
</div>

<div class="favs-grid">
  {% for r in favs %}
  <div class="favcell" data-rel="{{ r }}">
    <img src="{{ url_for('person_thumb', name=name, rel=r) }}" loading="lazy" alt="{{ r }}"
         title="{{ r }}" onclick="lbOpen({{ loop.index0 }})">
    <button class="unfav" title="Quitar de favoritos" onclick="unFav(this, {{ r|tojson }})">★</button>
  </div>
  {% endfor %}
  <a class="favadd" href="{{ url_for('person_page', name=name) }}" title="Las fotos se marcan con la ⭐ del visor">
    <span style="font-size:24px">＋</span><span>Agregar</span>
  </a>
</div>

<button id="fav-top" class="secondary iconbtn" onclick="scrollTo({top:0,behavior:'smooth'})"
        title="Subir al top" style="position:fixed; right:22px; bottom:22px; z-index:15;
        border-radius:50%; display:none; box-shadow:0 4px 14px rgba(0,0,0,.5)">↑</button>

<script>
window.LB_PHOTOS = [
  {% for r in favs %}{url:"{{ url_for('person_photo', name=name, rel=r) }}",name:"{{ r|e }}"},{% endfor %}
];
addEventListener('scroll', () => {
  document.getElementById('fav-top').style.display = scrollY > 400 ? 'flex' : 'none';
});
const FAV_URL = "{{ url_for('post_fav', name=name) }}";
function unFav(btn, rel) {
  fetch(FAV_URL, {method:'POST', headers:{'Content-Type':'application/x-www-form-urlencoded'},
    body: new URLSearchParams({rel: rel, fav: ''})}).then(r => r.json()).then(d => {
      if (!d.ok) return;
      btn.closest('.favcell').remove();
      document.getElementById('favcount').textContent =
        document.querySelectorAll('.favcell').length;
    });
}
</script>
__LIGHTBOX__
"""
FAVS_TEMPLATE = FAVS_TEMPLATE.replace("__LIGHTBOX__", LIGHTBOX)


@app.route("/person/<name>/favoritos")
def person_favs(name):
    """La página de favoritas de la persona: verlas todas, quitar, y el visor."""
    library = get_library()
    if not library or name not in load_known():
        return redirect(url_for("known_page"))
    rels = person_photos(library, name) + person_photos(library, name, only_nsfw=True)
    posts = posts_mod.get_many(name, rels)
    favs = [r for r in rels if posts[r]["fav"]]
    thumb = KNOWN_THUMBS / f"{safe_name(name)}.jpg"
    return render_template_string(
        shell("known", FAVS_TEMPLATE), name=name, favs=favs,
        has_thumb=thumb.is_file(),
        cache_bust=int(thumb.stat().st_mtime) if thumb.is_file() else 0)


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
<h1>Separar caras <span class="hint-ico" title="Encuentra las fotos donde hay más de una persona y las aparta a la subcarpeta {{ group_dir }}. ¿Para qué? El importador aprende de cada foto la cara más grande: eso es infalible si la persona está sola, pero en una foto grupal puede aprender la cara equivocada. Separando las grupales, el import solo aprende de fotos seguras — y las grupales las procesás aparte con el corrector, que pregunta cuál cara es cuál. Ojo con el tamaño mínimo: solo se cuentan las caras grandes. Así una selfie en la calle con transeúntes de fondo no se toma como foto grupal.">ⓘ</span></h1>

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
        data-confirm="Se moverán las fotos tildadas a {{ group_dir }} dentro de la carpeta revisada. ¿Continuar?">
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
<div class="panel"><span class="hint-ico" title="✅ No hay fotos con más de una cara (con ese tamaño mínimo). Podés importar esta carpeta sin riesgo.">ⓘ</span></div>
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


def _dupes_worker(folder, threshold, exclude, video_max_mb):
    def progress(current, total, name):
        DUPE_STATE.update(current=current, total=total, photo=name)
    try:
        find_duplicates(folder, threshold=threshold, exclude=exclude, progress=progress,
                        video_max_mb=video_max_mb)
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
    cfg = load_config()
    video_max_mb = cfg.get("dup_video_max_mb", DEFAULT_VIDEO_MAX_MB)
    DUPE_STATE.update(status="running", current=0, total=0, photo="", error="",
                      folder=folder)
    threading.Thread(target=_dupes_worker,
                     args=(Path(folder), DEFAULT_THRESHOLD, exclude, video_max_mb),
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


PORT = 5000


def _matar_fantasma_en_puerto(port):
    """Si quedó un proceso viejo ESCUCHANDO en `port` (un fantasma de una ventana
    anterior que no cerró bien), lo cierra antes de arrancar. Así el arranque no
    falla en silencio ni sirve una versión vieja. Solo Windows (netstat/taskkill),
    y nunca se mata a sí mismo."""
    import subprocess
    try:
        out = subprocess.run(["netstat", "-ano"], capture_output=True, text=True,
                             timeout=5).stdout
    except Exception:
        return
    mi_pid = str(os.getpid())
    pids = set()
    for line in out.splitlines():
        parts = line.split()
        # TCP | Dirección local | Dirección remota | ESTADO | PID
        if (len(parts) >= 5 and parts[0].upper().startswith("TCP")
                and parts[3].upper() == "LISTENING"
                and parts[1].endswith(f":{port}")):
            pid = parts[-1]
            if pid.isdigit() and pid not in (mi_pid, "0"):
                pids.add(pid)
    for pid in pids:
        try:
            subprocess.run(["taskkill", "/F", "/PID", pid], capture_output=True, timeout=5)
            print(f"[web] Cerré un proceso viejo que ocupaba el puerto {port} (PID {pid}).")
        except Exception:
            pass


def _limpiar_huerfanos():
    """Saca de data/ la metadata de carpetas que ya no existen (ver limpieza.py).
    Nunca debe impedir que la app arranque: si falla, se avisa y se sigue."""
    try:
        from app import limpieza
        print(limpieza.resumen(limpieza.purgar(get_library())))
    except Exception as e:
        print(f"[limpieza] no se pudo: {e}")


def _modelos_para_calentar():
    """Los modelos que la app usa (fiado de configuración + constantes), sin
    repetidos. El precarga (vlm.calentar) deja el último residente en la GPU,
    así que se ordena para que el VLM (etiquetas/narración/copetes) quede
    arriba cuando la VRAM (16 GB) obligue a elegir uno grande a la vez."""
    modelos = []
    try:
        cfg = load_config()
        for k in ("pie_model", "comentarios_model", "narracion_model",
                  "post_model", "copete_model"):
            m = (cfg.get(k) or "").strip()
            if m:
                modelos.append(m)
    except Exception:
        pass
    modelos.append(COPETE_MODEL)
    try:
        modelos.append(vlm.tag_model())
    except Exception:
        pass
    return modelos


def _precalentar_modelos():
    """Carga en la GPU los modelos de IA al arrancar, en un hilo de fondo, para
    que el primer «crear contenido» no pague la carga de 7-13s por foto."""
    def job():
        try:
            calentados = vlm.calentar(_modelos_para_calentar())
            if calentados:
                print(f"[ia] Modelos cargados en GPU: {', '.join(calentados)}")
        except Exception as e:
            print(f"[ia] Precarga de modelos falló (no es grave): {e}")
    threading.Thread(target=job, daemon=True).start()


def _log_crash(etype, value, tb):
    """Si el servidor se corta con una excepción, dejar el traceback completo en
    data/crash.log para poder cazarla a la próxima (ej. el cierre al generar el
    pie de una foto). No corta el comportamiento normal: la consola igual muestra
    el error."""
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(DATA_DIR / "crash.log", "a", encoding="utf-8") as f:
            f.write(f"\n--- se cortó el servidor · {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
            traceback.print_exception(etype, value, tb, file=f)
    except Exception:
        pass
    sys.__excepthook__(etype, value, tb)


def _thread_crash(args):
    _log_crash(args.exc_type, args.exc_value, args.exc_traceback)


sys.excepthook = _log_crash
threading.excepthook = _thread_crash


def main():
    # El reloader corre DOS procesos (el vigía y el servidor de verdad), y los
    # dos pasan por acá: sin esta guarda se abrían dos pestañas. El navegador
    # lo abre solo el vigía (que arranca una única vez); el servidor, que se
    # reinicia con cada cambio de código, no abre nada.
    if os.environ.get("WERKZEUG_RUN_MAIN") != "true":
        # Al ABRIR (una sola vez, en el vigía — no en cada reinicio del reloader),
        # echar cualquier fantasma de una ventana anterior que haya quedado en el
        # puerto. Así se mantiene el auto-reinicio del reloader y el puerto siempre
        # queda libre para esta ventana.
        _matar_fantasma_en_puerto(PORT)
        # NO se purga huérfanos al arrancar. Se probó y fue un error grave: una
        # carpeta vacía un momento (o el disco sin montar del todo) hacía que se
        # borraran etiquetas reales sin que el usuario tocara nada. "Que vacíe si
        # vacío" era la acción MANUAL de vaciar, no un barrido automático. La
        # limpieza queda solo a pedido (botón).
        print(f"[web] Abrí http://127.0.0.1:{PORT}/inicio en el navegador (solo accesible desde esta máquina).")
        threading.Timer(1.5, lambda: webbrowser.open(f"http://127.0.0.1:{PORT}/inicio")).start()
    else:
        # El servidor de verdad (proceso del reloader), una vez por corrida:
        # deja los modelos de IA precargados en la GPU.
        _precalentar_modelos()
    # use_reloader=True: si se edita el código, el servidor se reinicia SOLO (cómodo
    # cuando Claude toca algo). El fantasma que eso puede dejar al cerrar la ventana
    # lo limpia _matar_fantasma_en_puerto en el próximo arranque.
    # threaded=True: el sugeridor en streaming tiene una respuesta abierta hasta
    # ~90s; sin hilos, el resto de la app (el sondeo del lote, las miniaturas)
    # quedaría bloqueado todo ese rato esperando que termine.
    app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=True, threaded=True)


if __name__ == "__main__":
    main()
