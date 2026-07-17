"""El "post" de cada foto: el pie y los comentarios, guardados en disco.

La galería muestra cada foto como si fuera una publicación: la foto, un pie
escrito por la IA, y unos comentarios de mentira firmados por seguidores
inventados.

Las dos cosas cuestan caro de generar —el modelo va a ~3,5 tok/s en CPU, o sea
casi dos minutos el pie— así que se hacen UNA vez por foto, a pedido, con un
botón, y quedan guardadas (data/posts.json). Al abrir la foto se muestran al
instante; si no gustaron, se borran y se regeneran.

Todo local: no sale nada de la máquina.
"""

import json
import re

from app import POSTS_PATH
from app import person_meta, vlm

# Tonos ofrecidos en Ajustes. La clave se guarda en la config; el texto es lo
# que se le manda al modelo (en español: lo entiende igual y sale más natural).
TONES = {
    "amistoso": "amistosos y cariñosos, como los de un amigo que te banca",
    "tibio": "tibios y educados, sin exagerar, como un conocido que no se juega",
    "picante": "picantes y directos, con doble sentido, subidos de tono",
    "hincha": "exagerados y fanáticos, como un hincha que te idolatra",
    "envidioso": "con un poco de envidia mal disimulada, filosos",
    "bobo": "bobos y random, como los comentarios sin sentido de internet",
}
DEFAULT_TONE = "amistoso"
N_COMMENTS = 3

# Los @usuarios NO los inventa el modelo: es un 4b y devuelve "@usuario1",
# "@usuario2", "@anónimo_3" (probado). Salen de esta lista, que el usuario edita
# en Ajustes. De paso el modelo escribe menos y tarda menos.
DEFAULT_USERS = [
    "sofi_ok", "juanma23", "laflor", "nacho.pd", "rominaaa", "elpela",
    "vale_ok", "tomi_87", "carito", "bruno.ok", "lulii", "fedeee",
    "agusdlp", "mica_ok", "santi.99", "pauli", "gonza_ok", "dai_",
]

# Ojo: NO poner comentarios de ejemplo acá. Probado: el modelo los copia textual
# en vez de inspirarse en ellos. El formato se describe, no se ejemplifica.
_PROMPT = """Mirá esta foto y escribí {n} comentarios distintos, como los de
seguidores en una red social. El tono tiene que ser: {tono}.

Reglas estrictas:
- Español rioplatense, de vos.
- MÁXIMO 10 PALABRAS cada uno. Son comentarios, no párrafos.
- Los {n} tienen que ser distintos entre sí y hablar de cosas distintas.
- Comentá lo que SE VE en la foto. Si no ves algo, no lo inventes.
- Escribí uno por línea, y NADA más: sin firmas, sin nombres, sin numerar,
  sin comillas, sin encabezado, sin explicar nada.
"""


def _load():
    if POSTS_PATH.is_file():
        try:
            return json.loads(POSTS_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save(data):
    POSTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    POSTS_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=1),
                          encoding="utf-8")


def _key(name, rel):
    return f"{name}/{rel}"


def get(name, rel):
    """El post de una foto: {"caption": str|None, "comments": [...]}.

    Siempre devuelve el dict completo, aunque la foto no tenga nada todavía."""
    got = _load().get(_key(name, rel)) or {}
    return {"caption": got.get("caption"), "comments": got.get("comments", [])}


def get_many(name, rels):
    """Los posts de varias fotos de una, sin releer el archivo por cada una."""
    data = _load()
    out = {}
    for rel in rels:
        got = data.get(_key(name, rel)) or {}
        out[rel] = {"caption": got.get("caption"), "comments": got.get("comments", [])}
    return out


def _put(name, rel, **campos):
    data = _load()
    post = data.get(_key(name, rel)) or {}
    post.update(campos)
    data[_key(name, rel)] = post
    _save(data)
    return post


def set_caption(name, rel, text):
    """Guarda el pie (también sirve para reescribirlo a mano)."""
    return _put(name, rel, caption=text)["caption"]


def set_comments(name, rel, comments):
    return _put(name, rel, comments=comments)["comments"]


def clear(name, rel):
    """Borra el post entero (pie + comentarios) de una foto."""
    data = _load()
    if data.pop(_key(name, rel), None) is not None:
        _save(data)


def clear_comments(name, rel):
    """Borra solo los comentarios, para regenerarlos sin perder el pie."""
    data = _load()
    post = data.get(_key(name, rel))
    if post and post.pop("comments", None) is not None:
        if post:
            data[_key(name, rel)] = post
        else:
            data.pop(_key(name, rel))
        _save(data)


def move_photo(name, old_rel, new_rel):
    """Si la foto se mueve, su post la sigue."""
    data = _load()
    got = data.pop(_key(name, old_rel), None)
    if got is not None:
        data[_key(name, new_rel)] = got
        _save(data)


def forget_person(name):
    """Se borró la persona: sus posts se van con ella."""
    data = _load()
    fuera = [k for k in data if k.startswith(f"{name}/")]
    if fuera:
        for k in fuera:
            data.pop(k)
        _save(data)


def rename_person(old, new):
    data = _load()
    movidos = {k: v for k, v in data.items() if k.startswith(f"{old}/")}
    if movidos:
        for k, v in movidos.items():
            data.pop(k)
            data[f"{new}/{k.split('/', 1)[1]}"] = v
        _save(data)


# --- generación (lenta: es el modelo de visión) ---------------------------

def generate_caption(name, rel, image_path, model=None):
    """Escribe el pie de foto y lo guarda. ~1,5 minutos en CPU.

    Sabemos de quién es la foto (sale de la carpeta), así que el pie lo escribe
    ELLA, en primera persona y con SU voz (la que configuraste en su perfil)."""
    raw = vlm.describe(image_path, model=model or vlm.DEFAULT_MODEL,
                       prompt=vlm.caption_prompt(name, person_meta.get_voice(name)),
                       prefill=vlm.PREFILL_PIE)
    text = vlm.clean_caption(raw)
    if not text:
        raise RuntimeError("El modelo devolvió un pie vacío.")
    _put(name, rel, caption=text)
    return text


def _parse(raw):
    """Saca los comentarios del texto crudo del modelo.

    Es un 4b y escribe torcido: numera, mete viñetas, comillas, se firma solo
    igual, o agrega un "Acá van:" al principio. Se limpia todo eso y se tiran
    las líneas que claramente no son un comentario (las largas y las de relleno).
    """
    out = []
    for line in raw.splitlines():
        t = line.strip()
        if not t:
            continue
        t = re.sub(r"^[-*•]\s*", "", t)              # viñetas
        t = re.sub(r"^\d+[.)]\s*", "", t)            # numeración
        t = re.sub(r"^@[\w.]{2,30}\s*:\s*", "", t)   # si se firmó solo, fuera
        t = t.strip().strip('"').strip("«»").strip()
        if not t:
            continue
        # relleno típico: "Acá van los comentarios:", "Comentarios:", etc.
        if t.endswith(":") or len(t.split()) > 25:
            continue
        out.append(t)
    return out


def _with_users(texts, users=None):
    """Le pone un @usuario distinto a cada comentario, salteado por la lista."""
    pool = list(users or DEFAULT_USERS) or list(DEFAULT_USERS)
    # Sin random: el orden depende del texto, así los mismos comentarios dan
    # siempre los mismos nombres (regenerar no baraja todo de nuevo).
    out, used = [], set()
    for i, t in enumerate(texts):
        j = (hash(t) + i) % len(pool)
        while pool[j] in used and len(used) < len(pool):
            j = (j + 1) % len(pool)
        used.add(pool[j])
        out.append({"user": pool[j], "text": t})
    return out


def generate_comments(name, rel, image_path, tone=DEFAULT_TONE, model=None,
                      n=N_COMMENTS, users=None):
    """Escribe los comentarios y los guarda. ~30s en CPU.

    Lanza RuntimeError con mensaje legible si Ollama no está, el modelo no existe
    o lo que devolvió no sirve."""
    prompt = _PROMPT.format(n=n, tono=TONES.get(tone, TONES[DEFAULT_TONE]))
    # system="": el de vlm.py es para el pie de foto (un párrafo, primera
    # persona) y acá lo pelearía. Los comentarios se guían solos con el prompt.
    raw = vlm.describe(image_path, model=model or vlm.DEFAULT_MODEL,
                       prompt=prompt, num_predict=45 * n, temperature=0.95,
                       system="")
    got = _parse(raw)
    if not got:
        raise RuntimeError(f"El modelo no devolvió comentarios usables: {raw[:120]}")
    con_firma = _with_users(got[:n], users)
    _put(name, rel, comments=con_firma)
    return con_firma
