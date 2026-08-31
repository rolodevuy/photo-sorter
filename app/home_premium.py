# -*- coding: utf-8 -*-
"""Home premium: portada editorial / biblioteca fotografica.

Modulo auxiliar que genera el contexto de datos para la portada (`inicio`).
Todo se arma con datos REALES de la biblioteca: conteo de fotos por persona,
fechas por mtime, favoritas del pie, y agregaciones por mes. Nada inventado.
"""
import random
import datetime
import os
import hashlib

from .known import known_summary, person_photos
from . import posts as posts_mod

_MESES_ES = ["", "enero", "febrero", "marzo", "abril", "mayo", "junio",
             "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"]
_DIA_ES = ["lunes", "martes", "miercoles", "jueves", "viernes", "sabado", "domingo"]


def _fecha_corta(ts):
    import datetime as _dt
    d = _dt.datetime.fromtimestamp(ts).date()
    hoy = _dt.date.today()
    dif = (hoy - d).days
    if dif <= 0:
        return "Hoy"
    if dif == 1:
        return "Ayer"
    return f"{d.day} {_MESES_ES[d.month][:3]}"


def _nfs(rel):
    return rel.replace("\\", "/")


def _photofiles(library, name):
    try:
        rels = person_photos(library, name)
    except Exception:
        return []
    out = []
    base = os.path.join(library, name)
    for r in rels:
        p = os.path.join(base, r)
        try:
            out.append((r, os.path.getmtime(p)))
        except OSError:
            continue
    return out


def _build_archivo(library, gente, limite=240):
    filas = []
    for g in gente:
        n = g["name"]
        try:
            for rel, mt in _photofiles(library, n):
                t = datetime.date.fromtimestamp(mt)
                filas.append((t.year, t.month, t.day, n, rel, mt))
        except Exception:
            continue
    if not filas:
        return []
    random.seed(hashlib.sha1(repr(sorted(n["name"] for n in gente)).encode("utf-8")).hexdigest())
    if len(filas) > limite:
        filas = random.sample(filas, limite)
    filas.sort(key=lambda x: x[5], reverse=True)
    return filas


def build_context(library):
    random.seed()
    ctx = {"library": bool(library)}
    try:
        gente = known_summary(library)
    except Exception:
        gente = []
    ctx["gente"] = gente

    def _full(name, rel):
        return f"/person_photo/{name}/{_nfs(rel)}"

    def _thumb(name, rel):
        return f"/person_thumb/{name}/{_nfs(rel)}"

    def _perfil(name):
        return f"/person/{name}"

    archivo = _build_archivo(library or "", gente)

    posts = {}
    try:
        posts = posts_mod._load()
    except Exception:
        posts = {}

    def _cap(name, rel):
        return (posts.get(f"{name}/{rel}", {}) or {}).get("caption")

    # Foto del dia (estable por fecha)
    foto_dia = None
    if archivo:
        hoy = datetime.date.today()
        sem = hashlib.sha1(f"fotodia:{hoy.strftime('%Y%m%d')}".encode()).hexdigest()
        idx = int(sem, 16) % len(archivo)
        y, m, d, name, rel, mt = archivo[idx]
        foto_dia = {"url": _full(name, rel), "url_full": _full(name, rel),
                    "name": name, "caption": _cap(name, rel) or None}
    ctx["foto_dia"] = foto_dia

    # Hero: favoritas reales + respaldo SFW
    favs = [k.split("/", 1) for k, v in posts.items()
            if (v or {}).get("fav") and "/" in k and k.count("/") >= 1]
    hero_items = []
    if favs:
        rnd = random.Random()
        rnd.shuffle(favs)
        for name, rel in favs[:5]:
            mt = 0
            try:
                mt = os.path.getmtime(os.path.join(library or "", name, rel.replace("/", os.sep)))
            except OSError:
                pass
            hero_items.append({"name": name, "rel": rel, "mt": mt})
    else:
        seen = set()
        no_hero = 0
        r = random.Random()
        while no_hero < 40 and len(hero_items) < 5 and archivo:
            no_hero += 1
            y, m, d, name, rel, mt = r.choice(archivo)
            if (name, rel) in seen:
                continue
            seen.add((name, rel))
            hero_items.append({"name": name, "rel": rel, "mt": mt})
    for it in hero_items:
        rel = it["rel"]
        cap = _cap(it["name"], rel)
        it["titulo"] = cap[:54] if cap else it["name"]
        it["kicker"] = "Foto de biblioteca"
        mt = it.get("mt") or 0
        it["chips"] = []
        if mt:
            it["chips"].append(_fecha_corta(mt))
        if (posts.get(f"{it['name']}/{rel}", {}) or {}).get("fav"):
            it["chips"].append("Favorita")
        it["meta"] = it["name"]
        it["perfil"] = _perfil(it["name"])
        it["url"] = _thumb(it["name"], rel)
        it["url_full"] = _full(it["name"], rel)
    ctx["hero_items"] = hero_items

    # Protagonistas (mas fotos SFW reales)
    protagonistas = []
    try:
        filas = _build_archivo(library or "", gente, limite=8000)
    except Exception:
        filas = []
    conteo = {}
    muestra = {}
    for y, m, d, name, rel, mt in filas:
        conteo[name] = conteo.get(name, 0) + 1
        if name not in muestra:
            muestra[name] = (rel, mt)
    for name, c in sorted(conteo.items(), key=lambda kv: (-kv[1], kv[0])):
        rel, mt = muestra[name]
        protagonistas.append({"name": name, "fotos": c,
                              "img": _thumb(name, rel), "perfil": _perfil(name)})
    ctx["protagonistas"] = protagonistas

    # Recien agregadas (mtime mas reciente)
    recien = []
    deps = set()
    for y, m, d, name, rel, mt in archivo:
        if name in deps:
            continue
        deps.add(name)
        recien.append({"name": name, "thumb": _thumb(name, rel), "mt": mt,
                       "url_full": _full(name, rel), "fecha": _fecha_corta(mt),
                       "perfil": _perfil(name)})
    recien.sort(key=lambda x: x["mt"], reverse=True)
    ctx["recien"] = recien[:9]

    # En este dia (mtime mes+dia = hoy)
    hoy = datetime.date.today()
    en_este_dia = []
    for y, m, d, name, rel, mt in archivo:
        if m == hoy.month and d == hoy.day:
            en_este_dia.append({"name": name, "thumb": _thumb(name, rel),
                                "url_full": _full(name, rel),
                                "etiqueta": f"{d} de {_MESES_ES[m]}",
                                "perfil": _perfil(name)})
        if len(en_este_dia) >= 6:
            break
    ctx["en_este_dia"] = en_este_dia

    # Colecciones: por mes + Favoritas
    meses = {}
    for y, m, d, name, rel, mt in archivo:
        k = (y, m)
        meses.setdefault(k, []).append((name, rel, mt))
    meses_items = []
    for (y, m), items in sorted(meses.items(), reverse=True)[:8]:
        items.sort(key=lambda x: x[2], reverse=True)
        nn, rr, mt = items[0]
        meses_items.append({"titulo": f"{_MESES_ES[m].capitalize()} {y}",
                            "n": len(items), "img": _thumb(nn, rr), "url": "/known"})
    favoritas = None
    if favs:
        ff = favs[0]
        favoritas = {"titulo": "Favoritas", "n": len(favs),
                     "img": _thumb(ff[0], ff[1]), "url": "/known"}
    ctx["colecciones"] = {"meses": meses_items, "favoritas": favoritas}

    # Descubrimientos (curiosidades reales)
    desc = []
    if protagonistas:
        top = protagonistas[0]
        desc.append({"emoji": "\U0001F451", "url": _perfil(top["name"]),
                     "texto": f"{top['name']} es quien mas aparece: {top['fotos']} fotos."})
    if meses_items:
        big = max(meses_items, key=lambda c: c["n"])
        desc.append({"emoji": "\U0001F4C5", "url": big["url"],
                     "texto": f"{big['titulo']} fue tu mes con mas recuerdos ({big['n']} fotos)."})
    if len(protagonistas) >= 2:
        seg = protagonistas[1]
        desc.append({"emoji": "\U0001F3AC", "url": _perfil(seg["name"]),
                     "texto": f"Tras de {protagonistas[0]['name']}, {seg['name']} es la segunda cara que mas aparece."})
    if en_este_dia:
        desc.append({"emoji": "\U0001F550", "url": "/known",
                     "texto": f"Hay {len(en_este_dia)} recuerdo(s) marcados para el dia de hoy."})
    ctx["descubrimientos"] = desc

    return ctx


HOME_TEMPLATE = r"""
<style>
  /* ── Home premium: fototeca editorial ─────────────────────────────────────
     Álbum fotográfico personal + biblioteca digital. La foto manda, sobra
     aire, tipografía cuidada, tarjetas grandes. Paleta oscura de la app con
     acentos cálidos para las marcas editoriales. */

  .ph-head { display:flex; align-items:center; gap:1.4rem; padding:0 0 1.6rem;
    border-bottom:1px solid var(--bd); margin-bottom:2.6rem; flex-wrap:wrap; }
  .ph-brand { display:flex; align-items:center; gap:.75rem; }
  .ph-mark { width:44px; height:44px; border-radius:12px; overflow:hidden; flex:none;
    border:1px solid var(--bd); background:var(--panel2); display:flex;
    align-items:center; justify-content:center; font-size:1.4rem; box-shadow:0 6px 18px rgba(0,0,0,.25); }
  .ph-word .ph-name { font-size:1.18rem; font-weight:760; letter-spacing:-.01em; line-height:1.05; color:var(--tx); }
  .ph-word .ph-sub { font-size:.7rem; color:var(--mut); letter-spacing:.06em; text-transform:uppercase; }
  .ph-nav { display:flex; gap:.2rem; margin-left:.6rem; }
  .ph-nav a { padding:.5rem .85rem; border-radius:999px; color:var(--mut); font-size:.92rem;
    font-weight:600; text-decoration:none; transition:color .18s, background .18s; }
  .ph-nav a:hover { color:var(--tx); background:var(--panel2); }
  .ph-right { margin-left:auto; display:flex; align-items:center; gap:.55rem; }
  .ph-search { position:relative; display:flex; align-items:center; }
  .ph-search .tic { position:absolute; left:.85rem; color:var(--mut); font-size:.95rem; pointer-events:none; }
  .ph-search input { background:var(--panel); border:1px solid var(--bd); color:var(--tx);
    border-radius:999px; padding:.6rem 1rem .6rem 2.2rem; font-size:.9rem; width:min(260px,34vw);
    outline:none; transition:border-color .18s, box-shadow .18s; }
  .ph-search input:focus { border-color:var(--acc2); box-shadow:0 0 0 3px rgba(88,166,255,.15); }
  .ph-av { width:40px; height:40px; border-radius:50%; border:1px solid var(--bd);
    background:var(--panel2); color:var(--tx); font-size:1.1rem; cursor:pointer;
    display:flex; align-items:center; justify-content:center; text-decoration:none; transition:transform .18s, border-color .18s; }
  .ph-av:hover { transform:scale(1.06); border-color:var(--acc2); }

  .sec-kick { display:flex; align-items:center; gap:.6rem; margin:0 0 .7rem;
    font-size:.72rem; font-weight:750; letter-spacing:.14em; text-transform:uppercase; color:var(--acc2); }
  .sec-kick::after { content:""; height:1px; background:var(--bd); flex:1; }
  .sec-title { font-size:1.7rem; font-weight:760; letter-spacing:-.02em; color:var(--tx); line-height:1.1; margin:0; }
  body .ph-sec { margin-top:3.6rem; }
  .ph-sec .sub { color:var(--mut); font-size:.95rem; margin:.5rem 0 0; }
  a.sec-mas { display:inline-flex; align-items:center; gap:.35rem; font-size:.88rem; font-weight:650;
    color:var(--acc2); text-decoration:none; margin-top:.6rem; }
  a.sec-mas:hover { color:#79c0ff; }

  /* HERO */
  .ph-hero { position:relative; border-radius:18px; overflow:hidden; border:1px solid var(--bd);
    background:#000; box-shadow:0 20px 50px rgba(0,0,0,.35); }
  .hero-slides { display:flex; transition:transform .6s cubic-bezier(.4,0,.2,1); }
  .hero-slide { flex:0 0 100%; position:relative; aspect-ratio:2.6/1; min-height:300px; }
  .hero-slide img { position:absolute; inset:0; width:100%; height:100%; object-fit:cover; display:block; }
  .hero-slide .ov { position:absolute; inset:0; background:linear-gradient(180deg,rgba(0,0,0,0) 55%,rgba(0,0,0,.72) 100%); }
  .hero-slide .hero-nofoto { position:absolute; inset:0; display:flex; align-items:center;
    justify-content:center; font-size:4rem; color:var(--mut); }
  .hero-txt { position:absolute; left:0; right:0; bottom:0; padding:2rem 2.2rem;
    display:flex; align-items:flex-end; justify-content:space-between; gap:1.2rem; }
  .hero-txt .ht-k { font-size:.78rem; font-weight:700; letter-spacing:.12em; text-transform:uppercase;
    color:#c9d1d9; margin-bottom:.55rem; }
  .hero-txt .ht-t { font-size:2.3rem; font-weight:740; letter-spacing:-.02em; color:#fff; line-height:1.08;
    text-shadow:0 2px 16px rgba(0,0,0,.6); max-width:32ch; }
  .hero-txt .ht-m { color:#d7dbe3; font-size:.92rem; margin-top:.5rem; display:flex; gap:.5rem; flex-wrap:wrap; }
  .hero-txt .ht-chip { display:inline-flex; align-items:center; gap:.35rem; background:rgba(13,17,23,.5);
    border:1px solid rgba(255,255,255,.18); backdrop-filter:blur(6px); border-radius:999px;
    padding:.28rem .7rem; font-size:.8rem; color:#e6edf3; }
  a.ph-hero-cta { display:inline-flex; align-items:center; gap:.4rem; background:#fff; color:#0d1117;
    font-weight:700; font-size:.9rem; padding:.6rem 1.2rem; border-radius:999px; text-decoration:none;
    transition:transform .18s, background .18s; flex:none; }
  a.ph-hero-cta:hover { background:#e6edf3; transform:translateY(-1px); }
  .hero-nav { position:absolute; top:50%; transform:translateY(-50%); z-index:3; width:2.7rem; height:2.7rem;
    border-radius:50%; border:1px solid rgba(255,255,255,.3); background:rgba(0,0,0,.35); color:#fff;
    font-size:1.4rem; cursor:pointer; display:flex; align-items:center; justify-content:center;
    backdrop-filter:blur(6px); transition:background .18s; }
  .hero-nav:hover { background:rgba(0,0,0,.6); }
  .hero-prev { left:.9rem; } .hero-next { right:.9rem; }
  .hero-dots { position:absolute; right:1.4rem; bottom:1.1rem; z-index:3; display:flex; gap:.38rem; }
  .hero-dot { width:20px; height:4px; border-radius:2px; background:rgba(255,255,255,.35); border:0;
    cursor:pointer; padding:0; transition:width .2s, background .2s; }
  .hero-dot.on { width:32px; background:#fff; }

  /* FOTO DEL DÍA + protagonistas */
  .ph-dia-num { display:flex; align-items:center; gap:2.6rem; }
  .ph-dia-card { position:relative; border-radius:16px; overflow:hidden; border:1px solid var(--bd);
    flex:0 0 46%; aspect-ratio:4/3; box-shadow:0 14px 34px rgba(0,0,0,.3); cursor:pointer; min-width:0; }
  .ph-dia-card img { position:absolute; inset:0; width:100%; height:100%; object-fit:cover; display:block;
    transition:transform .5s ease; }
  .ph-dia-card:hover img { transform:scale(1.03); }
  .ph-dia-card .ov { position:absolute; inset:0; background:linear-gradient(180deg,rgba(0,0,0,0) 50%,rgba(0,0,0,.7) 100%); }
  .ph-dia-card .dia-c { position:absolute; left:1rem; bottom:1rem; color:#fff; }
  .ph-dia-card .dia-tag { display:inline-block; font-size:.68rem; font-weight:750; letter-spacing:.1em;
    text-transform:uppercase; color:#fff; background:rgba(88,166,255,.85); border:none; border-radius:999px;
    padding:.22rem .6rem; margin-bottom:.5rem; }
  .ph-dia-card .dia-t { font-size:1.2rem; font-weight:700; letter-spacing:-.01em; }
  .ph-dia-card .dia-m { font-size:.82rem; color:#d7dbe3; margin-top:.2rem; }
  .ph-dia-lado { flex:1; min-width:0; }
  .ph-prota { display:grid; grid-template-columns:repeat(auto-fill,minmax(118px,1fr)); gap:1.1rem 1rem; margin-top:1.2rem; }
  .ph-prot { display:flex; flex-direction:column; align-items:center; text-align:center; text-decoration:none; color:inherit; }
  .ph-prot .pp-img { width:100%; aspect-ratio:1/1; border-radius:14px; overflow:hidden; border:1px solid var(--bd);
    background:var(--panel2); margin-bottom:.6rem; position:relative; box-shadow:0 8px 20px rgba(0,0,0,.2);
    transition:transform .2s, border-color .2s; }
  .ph-prot:hover .pp-img { transform:translateY(-3px); border-color:var(--acc2); }
  .ph-prot .pp-img img { width:100%; height:100%; object-fit:cover; display:block; }
  .ph-prot .pp-img .ppfb { position:absolute; inset:0; display:flex; align-items:center; justify-content:center; font-size:2rem; color:var(--mut); }
  .ph-prot .pp-n { font-weight:700; font-size:.86rem; color:var(--tx); line-height:1.15; max-width:100%;
    overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  .ph-prot .pp-c { font-size:.74rem; color:var(--mut); margin-top:.12rem; }

  /* Recién agregadas (masonry) */
  .ph-recien { display:grid; grid-template-columns:repeat(6,1fr); grid-auto-rows:130px; gap:.8rem; }
  .ph-rc { position:relative; border-radius:12px; overflow:hidden; border:1px solid var(--bd); cursor:pointer; background:var(--panel2); }
  .ph-rc.grid-c1 { grid-column:span 1; grid-row:span 1; }
  .ph-rc.grid-c2 { grid-column:span 2; grid-row:span 1; }
  .ph-rc.grid-r2 { grid-column:span 2; grid-row:span 2; }
  .ph-rc img { position:absolute; inset:0; width:100%; height:100%; object-fit:cover; display:block; transition:transform .5s ease; }
  .ph-rc:hover img { transform:scale(1.04); }
  .ph-rc .ov { position:absolute; inset:0; background:linear-gradient(180deg,rgba(0,0,0,0) 55%,rgba(0,0,0,.62) 100%); opacity:0; transition:opacity .25s; }
  .ph-rc:hover .ov { opacity:1; }
  .ph-rc .rc-m { position:absolute; left:.6rem; bottom:.55rem; color:#fff; font-size:.72rem; font-weight:650;
    opacity:0; transform:translateY(3px); transition:opacity .25s, transform .25s; }
  .ph-rc:hover .rc-m { opacity:1; transform:none; }

  /* En este día */
  .ph-dia-en { display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:.9rem; }
  .ph-en { position:relative; border-radius:14px; overflow:hidden; border:1px solid var(--bd);
    aspect-ratio:1/1; cursor:pointer; background:var(--panel2); }
  .ph-en img { position:absolute; inset:0; width:100%; height:100%; object-fit:cover; display:block; transition:transform .5s ease; }
  .ph-en:hover img { transform:scale(1.04); }
  .ph-en .ov { position:absolute; inset:0; background:linear-gradient(180deg,rgba(0,0,0,0) 45%,rgba(0,0,0,.68) 100%); }
  .ph-en .en-c { position:absolute; left:.8rem; bottom:.7rem; color:#fff; }
  .ph-en .en-k { font-size:.7rem; font-weight:750; letter-spacing:.08em; text-transform:uppercase; color:#f2cc60; }
  .ph-en .en-n { font-weight:700; font-size:.9rem; margin-top:.1rem; }

  /* Colecciones */
  .ph-cols { display:grid; grid-template-columns:repeat(auto-fill,minmax(200px,1fr)); gap:1.1rem; }
  .ph-col { position:relative; border-radius:16px; overflow:hidden; border:1px solid var(--bd);
    aspect-ratio:4/3; text-decoration:none; color:#fff; display:block; box-shadow:0 10px 26px rgba(0,0,0,.25); }
  .ph-col img { position:absolute; inset:0; width:100%; height:100%; object-fit:cover; display:block; transition:transform .5s ease; }
  .ph-col:hover img { transform:scale(1.05); }
  .ph-col .ov { position:absolute; inset:0; background:linear-gradient(180deg,rgba(0,0,0,.05) 35%,rgba(0,0,0,.75) 100%); }
  .ph-col .ph-col-c { position:absolute; left:1rem; right:1rem; bottom:.9rem; }
  .ph-col .ph-col-n { font-size:1.05rem; font-weight:700; letter-spacing:-.01em; }
  .ph-col .ph-col-m { font-size:.78rem; color:#d7dbe3; margin-top:.15rem; }

  /* Descubrimientos */
  .ph-desc { display:grid; grid-template-columns:repeat(auto-fit,minmax(240px,1fr)); gap:.9rem; }
  .ph-desc-item { display:flex; align-items:center; gap:.85rem; background:linear-gradient(180deg,var(--panel),var(--panel2));
    border:1px solid var(--bd); border-radius:14px; padding:1rem 1.1rem; text-decoration:none; color:var(--tx);
    transition:border-color .18s, transform .18s; }
  .ph-desc-item:hover { border-color:var(--acc2); transform:translateY(-2px); }
  .ph-desc-item .di-em { flex:none; width:40px; height:40px; border-radius:12px; background:rgba(88,166,255,.12);
    display:flex; align-items:center; justify-content:center; font-size:1.3rem; }
  .ph-desc-item .di-t { font-size:.9rem; line-height:1.35; font-weight:550; }

  @media (max-width:900px) {
    .ph-dia-num { flex-direction:column; align-items:stretch; gap:1.6rem; }
    .ph-dia-card { flex:1 1 100%; }
    .ph-recien { grid-template-columns:repeat(2,1fr); grid-auto-rows:140px; }
    .ph-nav { display:none; }
  }
  @media (max-width:1180px) and (min-width:901px) { .ph-recien { grid-template-columns:repeat(4,1fr); } }
  @media (max-width:720px) { .hero-txt .ht-t { font-size:1.5rem; } .hero-slide { aspect-ratio:4/3; } }
</style>

<h1 style="display:none">Inicio</h1>

<header class="ph-head">
  <div class="ph-brand">
    <span class="ph-mark">📷</span>
    <span class="ph-word"><span class="ph-name">photo-sorter</span>
      <span class="ph-sub">Tu archivo fotográfico</span></span>
  </div>
  <nav class="ph-nav">
    <a href="{{ url_for('known_page') }}">Biblioteca</a>
    <a href="{{ url_for('known_page') }}">Colecciones</a>
    <a href="{{ url_for('known_page') }}">Personas</a>
  </nav>
  <div class="ph-right">
    <form class="ph-search" action="{{ url_for('finder_page') }}" method="get">
      <span class="tic">🔍</span>
      <input name="q" placeholder="Buscar fotos, personas, lugares, fechas..." aria-label="Buscar">
    </form>
    <a class="ph-av" href="{{ url_for('settings_page') }}" title="Perfil">👤</a>
  </div>
</header>

{% if not library %}
<div class="panel">
  <p class="meta" style="margin:0">Todavía no hay Biblioteca. Configurá la carpeta
    donde vive tu archivo de fotos, analizá y empezá a descubrir tus recuerdos.</p>
  <a class="dash-cta" style="margin-top:.7rem" href="{{ url_for('settings_page') }}">⚙ Configurar Biblioteca</a>
</div>

{% elif gente %}

{# HERO #}
{% if hero_items %}
<div class="ph-hero" id="hero">
  <div class="hero-slides" id="hero-slides">
    {% for it in hero_items %}
    <div class="hero-slide"
         onclick="abrirFoto('{{ it.url_full }}','{{ it.titulo }}','{{ it.meta }}','{{ it.perfil }}')">
      <img src="{{ it.url }}" alt="" loading="lazy"
           onerror="this.style.display='none';this.nextElementSibling.style.display='flex';">
      <div class="hero-nofoto" style="display:none">📷</div>
      <div class="ov"></div>
      <div class="hero-txt">
        <div>
          <div class="ht-k">{{ it.kicker }}</div>
          <div class="ht-t">{{ it.titulo }}</div>
          <div class="ht-m">
            {% for chip in it.chips %}<span class="ht-chip">{{ chip }}</span>{% endfor %}
          </div>
        </div>
        <a class="ph-hero-cta" href="{{ it.perfil }}">Ver colección →</a>
      </div>
    </div>
    {% endfor %}
  </div>
  <button class="hero-nav hero-prev" onclick="heroGo(-1)" aria-label="Anterior">‹</button>
  <button class="hero-nav hero-next" onclick="heroGo(1)" aria-label="Siguiente">›</button>
  <div class="hero-dots" id="hero-dots"></div>
</div>
{% endif %}

{# FOTO DEL DÍA + protagonistas #}
<section class="ph-sec">
  <div class="ph-dia-num">
    {% if foto_dia %}
    <div class="ph-dia-card"
         onclick="abrirFoto('{{ foto_dia.url_full }}','{{ foto_dia.name }} · {% if foto_dia.caption %}{{ foto_dia.caption[:70]|e }}{% else %}Foto del día{% endif %}','{{ url_for('person_page', name=foto_dia.name) }}')">
      <img src="{{ foto_dia.url }}" alt="" loading="lazy"
           onerror="this.style.display='none';this.nextElementSibling.style.display='flex';">
      <div class="hero-nofoto" style="display:none">📷</div>
      <div class="ov"></div>
      <div class="dia-c">
        <span class="dia-tag">Foto del día</span>
        <div class="dia-t">{% if foto_dia.caption %}{{ foto_dia.caption[:48] }}{% else %}{{ foto_dia.name }}{% endif %}</div>
        <div class="dia-m">{{ foto_dia.name }} · tocar para verla</div>
      </div>
    </div>
    {% endif %}
    <div class="ph-dia-lado">
      <div class="sec-kick">Reconocimiento facial</div>
      <h2 class="sec-title">Protagonistas</h2>
      <p class="sub">Las personas que más aparecen en tu biblioteca.</p>
      {% if protagonistas %}
      <div class="ph-prota">
        {% for p in protagonistas[:8] %}
        <a class="ph-prot" href="{{ p.perfil }}">
          <span class="pp-img">
            {% if p.img %}<img src="{{ p.img }}" alt="" loading="lazy"
                 onerror="this.nextElementSibling.style.display='flex';">
            <span class="ppfb" style="display:none">👤</span>
            {% else %}<span class="ppfb">👤</span>{% endif %}
          </span>
          <span class="pp-n">{{ p.name }}</span>
          <span class="pp-c">{{ p.fotos }} fotos</span>
        </a>
        {% endfor %}
      </div>
      {% endif %}
    </div>
  </div>
</section>

{# RECIÉN AGREGADAS #}
{% if recien %}
<section class="ph-sec">
  <div class="sec-kick">Lo más nuevo</div>
  <h2 class="sec-title">Agregadas recientemente</h2>
  <div class="ph-recien" style="margin-top:1.3rem">
    {% for f in recien[:9] %}
    {% set cls = 'grid-r2' if loop.index == 1 else ('grid-c2' if loop.index in (2,5) else 'grid-c1') %}
    <div class="ph-rc {{ cls }}" role="button" tabindex="0"
         onclick="abrirFoto('{{ f.url_full }}','{{ f.name }}','recién agregada · {{ f.fecha }}','{{ f.perfil }}')">
      <img src="{{ f.thumb }}" alt="" loading="lazy"
           onerror="this.nextElementSibling.style.display='flex';">
      <span class="hero-nofoto" style="display:none">📷</span>
      <div class="ov"></div>
      <span class="rc-m">{{ f.fecha }} · {{ f.name }}</span>
    </div>
    {% endfor %}
  </div>
  <a class="sec-mas" href="{{ url_for('known_page') }}">Ver toda la galería →</a>
</section>
{% endif %}

{# EN ESTE DÍA #}
{% if en_este_dia %}
<section class="ph-sec">
  <div class="sec-kick">Recuerdos</div>
  <h2 class="sec-title">En este día</h2>
  <div class="ph-dia-en" style="margin-top:1.3rem">
    {% for e in en_este_dia %}
    <div class="ph-en" role="button" tabindex="0"
         onclick="abrirFoto('{{ e.url_full }}','{{ e.name }}','{{ e.etiqueta }}','{{ e.perfil }}')">
      <img src="{{ e.thumb }}" alt="" loading="lazy"
           onerror="this.nextElementSibling.style.display='flex';">
      <span class="hero-nofoto" style="display:none">📷</span>
      <div class="ov"></div>
      <div class="en-c"><div class="en-k">{{ e.etiqueta }}</div><div class="en-n">{{ e.name }}</div></div>
    </div>
    {% endfor %}
  </div>
</section>
{% endif %}

{# COLECCIONES #}
{% if colecciones.meses or colecciones.favoritas %}
<section class="ph-sec">
  <div class="sec-kick">Tu archivo</div>
  <h2 class="sec-title">Colecciones</h2>
  <p class="sub">Agrupadas por mes, armadas con fotos de tu biblioteca.</p>
  <div class="ph-cols" style="margin-top:1.3rem">
    {% if colecciones.favoritas %}
    <a class="ph-col" href="{{ colecciones.favoritas.url }}">
      <img src="{{ colecciones.favoritas.img }}" alt="" loading="lazy"
           onerror="this.nextElementSibling.style.display='flex';">
      <span class="hero-nofoto" style="display:none">📷</span>
      <div class="ov"></div>
      <div class="ph-col-c"><div class="ph-col-n">⭐ {{ colecciones.favoritas.titulo }}</div>
        <div class="ph-col-m">{{ colecciones.favoritas.n }} foto(s)</div></div>
    </a>
    {% endif %}
    {% for c in colecciones.meses %}
    <a class="ph-col" href="{{ c.url }}">
      <img src="{{ c.img }}" alt="" loading="lazy"
           onerror="this.nextElementSibling.style.display='flex';">
      <span class="hero-nofoto" style="display:none">📷</span>
      <div class="ov"></div>
      <div class="ph-col-c"><div class="ph-col-n">{{ c.titulo }}</div>
        <div class="ph-col-m">{{ c.n }} fotos</div></div>
    </a>
    {% endfor %}
  </div>
</section>
{% endif %}

{# DESCUBRIMIENTOS #}
{% if descubrimientos %}
<section class="ph-sec">
  <div class="sec-kick">Pequeñas curiosidades</div>
  <h2 class="sec-title">Descubrimientos</h2>
  <div class="ph-desc" style="margin-top:1.3rem">
    {% for d in descubrimientos %}
    <a class="ph-desc-item" href="{{ d.url }}">
      <span class="di-em">{{ d.emoji }}</span>
      <span class="di-t">{{ d.texto }}</span>
    </a>
    {% endfor %}
  </div>
</section>
{% endif %}

{% endif %}

<script>
(function() {
  var slides = document.querySelectorAll('#hero-slides .hero-slide');
  if (!slides.length) return;
  var n = slides.length, i = 0, timer = null;
  var dots = document.getElementById('hero-dots');
  slides.forEach(function(_, k) {
    var d = document.createElement('button');
    d.className = 'hero-dot' + (k === 0 ? ' on' : '');
    d.setAttribute('aria-label', 'Ir al ' + (k + 1));
    d.onclick = function() { go(k); restart(); };
    dots.appendChild(d);
  });
  function paint() {
    document.getElementById('hero-slides').style.transform = 'translateX(-' + (i * 100) + '%)';
    [].forEach.call(dots.children, function(d, k) { d.classList.toggle('on', k === i); });
  }
  function go(d) { i = (i + d + n) % n; paint(); }
  function restart() { clearInterval(timer); timer = setInterval(function() { go(1); }, 5600); }
  window.heroGo = go;
  var h = document.getElementById('hero');
  if (h) { h.addEventListener('mouseenter', function() { clearInterval(timer); }); h.addEventListener('mouseleave', restart); }
  restart();
})();
</script>

<div id="gmodal" onclick="gcerrar(event)">
  <div id="gmodal-caja" onclick="event.stopPropagation()">
    <span id="gmodal-x" title="Cerrar">✕</span>
    <img id="gmodal-img" src="" alt="foto">
    <div id="gmodal-pie">
      <div id="gmodal-meta"></div>
      <a id="gmodal-perfil" href="#">Ir al perfil de <span id="gmodal-nombre"></span> →</a>
    </div>
  </div>
</div>

<script>
var GM = {
  abrir: function(url, meta, perfil, nombre) {
    var img = document.getElementById('gmodal-img');
    img.src = url || '';
    document.getElementById('gmodal-meta').textContent = meta || '';
    var p = document.getElementById('gmodal-perfil');
    p.href = perfil || '#'; p.style.visibility = perfil ? 'visible' : 'hidden';
    document.getElementById('gmodal-nombre').textContent = nombre || '';
    document.getElementById('gmodal').style.display = 'flex';
  }
};
function abrirFoto(url, titulo, meta, perfil) {
  GM.abrir(url, (titulo ? titulo + ' · ' : '') + (meta || ''), perfil, titulo);
}
function gcerrar(ev) {
  if (ev && ev.target && ev.target.id !== 'gmodal') return;
  document.getElementById('gmodal').style.display = 'none';
  document.getElementById('gmodal-img').src = '';
}
document.addEventListener('keydown', function(e) { if (e.key === 'Escape') gcerrar(); });
</script>

"""
