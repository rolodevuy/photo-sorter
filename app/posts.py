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

import hashlib
import json
import re

from app import POSTS_PATH
from app import finder, person_meta, vlm
from app import tags as tags_mod

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
N_COMMENTS = 6

# La perilla de Ajustes: qué tan subidos son los comentarios y el pie. El índice
# se guarda en la config. Vale para los dos, así todo el post tiene la misma onda.
#
# OJO: de "explícito" para arriba, un modelo alineado (qwen3.5:4b) se pone
# pudoroso y te lava las frases. Para esos niveles hay que elegir el abliterado
# en el desplegable del modal, que no tiene filtros (aunque ve peor).
CRUDENESS = [
    ("sano", "Nada subido de tono. Como comentaría un familiar."),
    ("pícaro", "Alguna insinuación y doble sentido, sin decir nada explícito."),
    ("subido", "Se nota que la desean y lo dicen, sin vueltas ni eufemismos."),
    ("explícito", "Sin filtro: dicen qué le harían y con qué palabras. "
                  "Las cosas por su nombre, sin suavizar nada."),
    ("guarro", "Lo más sucio, directo y ordinario posible. Puteadas incluidas. "
               "Sin ningún límite ni pudor."),
]
CRUDENESS_DEFAULT = 1

# Cada tantas fotos, uno de los comentarios habla de la edad ("qué señora tan
# elegante", "qué niña tan bonita"). "A veces" no se le puede pedir al modelo:
# si se lo pedís, lo hace SIEMPRE. Se decide acá, por foto, y solo entonces se
# le suma la instrucción al prompt. 1 de cada 3.
AGE_EVERY = 3
_AGE_RULE = ("- UNO solo de los comentarios, no más, tiene que comentar la edad "
             "que aparenta: si es grande algo como \"qué señora tan elegante\", "
             "si es joven algo como \"qué niña tan bonita\". Los demás ni la "
             "mencionan.")

# Los comentadores. Cada uno es alguien, no un tono suelto: pedirle al modelo
# "6 comentarios amistosos" da 6 comentarios iguales y genéricos; pedirle uno por
# cada uno de estos da 6 voces distintas.
#
# El @usuario y su personalidad viajan juntos, así @lulii es siempre la misma.
# Si en Ajustes cambiás la lista de nombres, se usan los tuyos y las
# personalidades se reparten igual.
# El tercer campo dice si ese comentador DESEA a la persona. Los que sí quedan
# afuera cuando la persona está etiquetada como menor (ver identidad()).
COMMENTERS = [
    ("lulii", "la amiga del alma, te banca en todo y exagera de puro cariño", False),
    ("elpela", "el pesado que tira onda sin disimular, se le va la mano", True),
    ("rominaaa", "la envidiosa, tira un elogio con veneno adentro", False),
    ("tomi_87", "el tímido, se anima a poco pero se le nota que babea", True),
    ("fedeee", "el bobo, comenta cualquier pavada sin sentido", False),
    ("carito", "la que solo mira la ropa y la pilcha, nada más", False),
    ("gonza_ok", "el hincha, te idolatra como si fueras de otro planeta", False),
    ("nacho.pd", "el que se hace el gracioso con un chiste malo", False),
    ("sofi_ok", "la que te desea sin filtro y lo dice crudo, sin vueltas", True),
    ("juanma23", "el curioso, pregunta dónde es o qué estabas haciendo", False),
]

DEFAULT_USERS = [u for u, _, _ in COMMENTERS]

# De las etiquetas de la persona sacamos CÓMO referirse a ella. Salen de las que
# ponés de a muchas en la galería ("estas 60 son hombres").
_GENERO_F = {"mujer", "femenino", "femenina", "chica", "señora", "niña", "nena",
             "piba", "mina", "dama"}
_GENERO_M = {"hombre", "masculino", "varón", "varon", "chico", "señor", "niño",
             "nene", "pibe", "tipo"}
# Etiquetas que dicen que es menor. Con cualquiera de estas: la perilla de
# crudeza se ignora y los comentadores que desean quedan afuera.
_MENOR = {"niño", "niña", "nene", "nena", "bebé", "bebe", "chiquito", "chiquita",
          "criatura", "menor", "pibe", "piba", "adolescente", "nieto", "nieta",
          "hijo", "hija"}


def identidad(tags):
    """De las etiquetas de la persona: {"genero": "f"|"m"|None, "menor": bool}.

    Sirve para que los comentarios se refieran bien a la persona ("qué linda" vs
    "qué lindo") y, sobre todo, para NO tratar a un chico como a un adulto."""
    t = {re.sub(r"[^\wáéíóúñü]", "", x.lower()) for x in tags}
    genero = "f" if t & _GENERO_F else ("m" if t & _GENERO_M else None)
    return {"genero": genero, "menor": bool(t & _MENOR)}

# Un 4b escribe en español neutro por más que le pidas "rioplatense": no sabe qué
# es. Prohibirle las formas concretas SÍ funciona, y como es una regla de
# gramática y no un ejemplo, no la copia textual (los ejemplos sí los copiaba).
_VOSEO = ("Escribís de VOS, nunca de tú: \"tenés\" y no \"tienes\", \"sos\" y no "
          "\"eres\", \"mirá\" y no \"mira\", \"vos\" y no \"tú\", \"te queda\" y "
          "no \"te va\". Nada de \"ti\" ni de \"contigo\".")

_PROMPT = """Mirá esta foto. Estas {n} personas la están comentando en una red
social. Escribí el comentario de cada una, en su voz:

{quienes}

Reglas estrictas:
- {quien_es}
- {voseo}
- Cada uno comenta ALGO CONCRETO de lo que está pasando en la foto: la postura,
  lo que está haciendo, cómo está puesta, qué se le ve. Si está sentada de
  piernas abiertas, se comenta ESO y si les gusta o no. Está PROHIBIDO el elogio
  genérico que serviría para cualquier otra foto ("qué linda", "te queda
  increíble", "estás radiante"): eso no dice nada.
- Si en la foto no pasa nada, comentan que no pasa nada. No inventen.
- Que se note quién es cada uno: tienen que sonar DISTINTOS entre sí.
- MÁXIMO 10 PALABRAS cada uno. Son comentarios, no párrafos.
- El clima general es: {tono}
- Qué tan subidos: {crudeza}
{edad}- Un comentario por línea, en el mismo orden, y NADA más: sin numerar, sin
  poner quién lo dice, sin comillas, sin encabezado, sin explicar nada.
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


def _stable_hash(s):
    """Un número a partir de un texto, IGUAL en cada corrida.

    hash() de Python NO sirve acá: viene salteado por proceso (PYTHONHASHSEED),
    así que al reiniciar la app cambiaba a quién le tocaba el comentario de la
    edad y qué @usuario firmaba cada comentario."""
    return int(hashlib.md5(s.encode("utf-8")).hexdigest()[:8], 16)


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

def crudeness_text(level):
    """El texto del nivel de la perilla, acotado por si la config trae fruta."""
    try:
        i = int(level)
    except (TypeError, ValueError):
        i = CRUDENESS_DEFAULT
    return CRUDENESS[max(0, min(i, len(CRUDENESS) - 1))][1]


def generate_caption(name, rel, image_path, model=None,
                     crudeness=CRUDENESS_DEFAULT):
    """Escribe el pie de foto y lo guarda. ~1,5 minutos en CPU.

    Sabemos de quién es la foto (sale de la carpeta), así que el pie lo escribe
    ELLA, en primera persona y con SU voz (la que configuraste en su perfil).

    Y como además reconocemos caras, miramos quién MÁS está en la foto: así el
    pie dice "acá comiendo con Alberto" y no "con alguien". Cuesta décimas de
    segundo al lado del minuto y medio del modelo, así que va siempre."""
    try:
        con = finder.who_is_in(image_path, exclude=[name])
    except Exception:
        con = []          # sin base de caras o foto ilegible: el pie igual sale
    # el pie va en primera persona, así que el género importa: "estoy cansada"
    # o "cansado" sale de las etiquetas de la persona
    ident = identidad(tags_mod.get_person(name))
    if ident["menor"]:
        crudeness = 0
    voz = person_meta.get_voice(name)
    crudeza, quien_es = crudeness_text(crudeness), quien_es_pie(ident)
    raw = vlm.describe(image_path, model=model or vlm.DEFAULT_MODEL,
                       prompt=vlm.caption_prompt(name, voz, con=con, crudeza=crudeza,
                                                 quien_es=quien_es),
                       prefill=vlm.PREFILL_PIE)
    # El modelo repite lo que le pusiste en el prompt: real, con un pie "vulgar y
    # directo" devolvió "Acá en la playa, ser directo, vulgar...". Sabemos qué le
    # inyectamos, así que se puede sacar sin adivinar.
    text = _sin_autonombre(
        _sin_instrucciones(vlm.clean_caption(raw),
                           [voz, crudeza, quien_es, vlm.SYSTEM_PIE]),
        name)
    if not text:
        raise RuntimeError("El modelo solo repitió las instrucciones. Probá de nuevo.")
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
        if t.endswith(":"):
            continue
        if len(t.split()) > 25:
            # Puede ser un párrafo de relleno... o los comentarios TODOS pegados
            # en un renglón ("qué genia! te ves feliz. lindo el rojo."), que pasa
            # seguido. Se parte en frases y se rescatan las que parezcan
            # comentarios; si ninguna lo parece, era relleno y se tira.
            frases = [f.strip() for f in re.split(r"(?<=[.!?])\s+", t)]
            cortas = [f for f in frases if f and 2 <= len(f.split()) <= 14]
            if len(cortas) >= 2:
                out.extend(cortas)
            continue
        out.append(t)
    return out


# Lo que se le escapa al modelo aunque le prohíbas el tuteo (2 de cada 6 igual
# salen en neutro). Se corrige acá y sale gratis. Solo formas INEQUÍVOCAS: nada
# de "mira"->"mirá" ni "tu"->"tu", que también son terceras personas o posesivos
# y romperían frases buenas.
_TUTEO = [
    (r"\bcontigo\b", "con vos"), (r"\bpara ti\b", "para vos"),
    (r"\ba ti\b", "a vos"), (r"\bde ti\b", "de vos"), (r"\bcon ti\b", "con vos"),
    (r"\bti\b", "vos"), (r"\btú\b", "vos"),
    (r"\beres\b", "sos"), (r"\btienes\b", "tenés"), (r"\bpuedes\b", "podés"),
    (r"\bquieres\b", "querés"), (r"\bhaces\b", "hacés"), (r"\bsabes\b", "sabés"),
    (r"\bestás muy\b", "estás re"), (r"\boye\b", "che"),
]


def _rioplatense(text):
    """Le saca el tuteo que se le escapa al modelo. Conserva las mayúsculas."""
    for patron, cambio in _TUTEO:
        def _rep(m, c=cambio):
            return c.capitalize() if m.group(0)[:1].isupper() else c
        text = re.sub(patron, _rep, text, flags=re.IGNORECASE)
    return text


def _sin_autonombre(texto, name):
    """Saca el "con Agus" del pie de foto... de Agus.

    El pie lo escribe ella en primera persona, así que nombrarse a sí misma como
    acompañante siempre está mal. Pedírselo al prompt no alcanza (probado: falla
    una de cada dos), y acá es seguro arreglarlo: en SU pie, "con ella" no existe.
    """
    n = re.escape(name)
    # "con Agus y Alberto" -> "con Alberto" (si no, queda "Acá y Alberto")
    t = re.sub(rf"\b(con|junto a|al lado de)\s+{n}\s+y\s+", r"\1 ", texto, flags=re.I)
    t = re.sub(rf"[,;]?\s*\b(con|junto a|al lado de|y)\s+{n}\b", "", t, flags=re.I)
    t = re.sub(r"\s{2,}", " ", t)
    t = re.sub(r"\s+([,.;!?])", r"\1", t)          # " ," -> ","
    t = re.sub(r"^\W+", "", t)
    return t.strip()


def _palabras(s):
    return re.sub(r"[^\w\s]", " ", s.lower()).split()


def _ngramas(s, n=4):
    p = _palabras(s)
    return {" ".join(p[i:i + n]) for i in range(len(p) - n + 1)}


def eco_de(texto, fragmentos, n=4):
    """True si `texto` repite algo que le metimos nosotros en el prompt.

    Compara tiradas de n palabras seguidas: un eco tiene tramos largos idénticos,
    una coincidencia real no. Textos muy cortos no se juzgan (no tienen tirada)."""
    mios = set()
    for f in fragmentos:
        mios |= _ngramas(f, n)
    if not mios:
        return False
    return bool(_ngramas(texto, n) & mios)


def _sin_instrucciones(texto, fragmentos):
    """Saca del texto las frases donde el modelo repitió las instrucciones.

    Real: pidiéndole un pie "vulgar y directo" devolvió "Acá en la playa, ser
    directo, vulgar...". Va frase por frase para no tirar el pie entero cuando
    solo una parte es eco."""
    frases = [f.strip() for f in re.split(r"(?<=[.!?])\s+|\n+", texto) if f.strip()]
    quedan = [f for f in frases if not eco_de(f, fragmentos)]
    return " ".join(quedan).strip()


def _sin_eco(textos, personas):
    """Tira los "comentarios" que en realidad son la descripción del personaje.

    Probado: al listarle los personajes en el prompt, el modelo devuelve alguno
    tal cual ("el hincha, te idolatra como si fueras de otro planeta") en vez de
    escribir su comentario, o se lo pone de prefijo ("el curioso: dónde andás?").
    """
    def clave(s):
        return " ".join(_palabras(s))
    eco = {clave(p) for p in personas}
    # "el curioso: dónde andás?" -> "dónde andás?". El prefijo son las primeras
    # palabras de alguna personalidad, seguidas de : o -. Se prueban arranques de
    # 2 y 3 palabras porque el modelo corta la etiqueta donde se le canta ("el
    # curioso:" y no "el curioso pregunta:"). De más largo a más corto, para que
    # gane el match más específico.
    arranques = []
    for p in personas:
        w = _palabras(p)
        arranques += [" ".join(w[:k]) for k in (3, 2) if len(w) >= k]
    arranques = sorted(set(arranques), key=len, reverse=True)
    prefijo = re.compile(r"^\W*(?:%s)\b[^:\-–]{0,30}\s*[:\-–]\s*"
                         % "|".join(re.escape(a) for a in arranques),
                         re.I) if arranques else None
    fuera = []
    for t in textos:
        if prefijo:
            t = prefijo.sub("", t).strip()
        if not t:
            continue
        k = clave(t)
        # copia textual, el arranque de una, o una tirada larga en común
        if k in eco or any(e.startswith(k[:25]) for e in eco if len(k) > 12):
            continue
        if eco_de(t, personas):
            continue
        fuera.append(t)
    return fuera


def quien_es_texto(ident):
    """La regla de cómo referirse a la persona, para el prompt."""
    if ident["menor"]:
        # Es un chico: comentarios tiernos y nada más. Ver generate_comments,
        # que además saca a los comentadores que desean y apaga la crudeza.
        base = ("La persona de la foto es un CHICO. Los comentarios son cariñosos "
                "y tiernos, como los de una tía: \"qué lindo chiquito\", \"mirá "
                "lo grande que está\". Usá diminutivos (lindito, chiquito, "
                "grandote). NADA sexual, NADA subido de tono, NADA sobre su "
                "cuerpo: eso está terminantemente prohibido acá.")
        if ident["genero"]:
            base += (" Es una nena: hablá en femenino." if ident["genero"] == "f"
                     else " Es un nene: hablá en masculino.")
        return base
    if ident["genero"] == "f":
        return ("La persona de la foto es una MUJER: referite a ella en femenino "
                "(\"qué linda\", \"estás hermosa\"), nunca en masculino.")
    if ident["genero"] == "m":
        return ("La persona de la foto es un HOMBRE: referite a él en masculino "
                "(\"qué lindo\", \"estás hecho un toro\"), nunca en femenino.")
    return ("No sabemos si es hombre o mujer: no arriesgues el género, escribí "
            "los comentarios de forma que sirvan para cualquiera.")


def quien_es_pie(ident):
    """Igual que quien_es_texto pero para el pie, que va en PRIMERA persona.

    Sin esto sale "estoy cansado" cuando es una mujer."""
    if ident["menor"]:
        base = ("Sos un chico. El pie es inocente y tierno, de lo que estás "
                "haciendo y nada más. NADA sexual ni subido de tono, y nada "
                "sobre tu cuerpo: está terminantemente prohibido acá.")
        if ident["genero"] == "f":
            return base + " Sos nena: escribí en femenino (\"estoy cansada\")."
        if ident["genero"] == "m":
            return base + " Sos nene: escribí en masculino (\"estoy cansado\")."
        return base
    if ident["genero"] == "f":
        return "Sos mujer: escribí en femenino (\"estoy cansada\", \"tirada\")."
    if ident["genero"] == "m":
        return "Sos hombre: escribí en masculino (\"estoy cansado\", \"tirado\")."
    return ""


def _pick(name, rel, n, users=None, solo_sanos=False):
    """Qué comentadores comentan ESTA foto: n de los COMMENTERS.

    Estable por foto (ver _stable_hash) pero distinto entre fotos, así no
    comentan siempre los mismos seis. Si en Ajustes cambiaste los nombres, se
    usan los tuyos y las personalidades se reparten entre ellos.

    solo_sanos saca a los que desean: es para las fotos de chicos."""
    pool = [(u, p) for u, p, desea in COMMENTERS if not (solo_sanos and desea)]
    if users:
        pool = [(users[i % len(users)], p) for i, (_, p) in enumerate(pool)]
    arranque = _stable_hash(_key(name, rel)) % len(pool)
    return [pool[(arranque + i) % len(pool)] for i in range(min(n, len(pool)))]


def generate_comments(name, rel, image_path, tone=DEFAULT_TONE, model=None,
                      n=N_COMMENTS, users=None, crudeness=CRUDENESS_DEFAULT):
    """Escribe los comentarios y los guarda. ~1 minuto en CPU.

    Comenta un puñado de los COMMENTERS, elegidos según la foto: cada uno tiene
    su personalidad y su @usuario, así los seis suenan distinto entre sí.

    Lanza RuntimeError con mensaje legible si Ollama no está, el modelo no existe
    o lo que devolvió no sirve."""
    # Cómo referirse a la persona, según SUS etiquetas (las que ponés de a
    # muchas en la galería). Si dicen que es un chico, la perilla de crudeza no
    # corre y los que desean no comentan: no es negociable desde Ajustes.
    ident = identidad(tags_mod.get_person(name))
    if ident["menor"]:
        crudeness = 0
    elegidos = _pick(name, rel, n, users, solo_sanos=ident["menor"])
    quienes = "\n".join(f"{i + 1}. {p}" for i, (_, p) in enumerate(elegidos))
    # el de la edad, 1 de cada AGE_EVERY fotos: estable por foto, así regenerar
    # no lo hace aparecer y desaparecer
    edad = _AGE_RULE + "\n" if _stable_hash(_key(name, rel)) % AGE_EVERY == 0 else ""
    prompt = _PROMPT.format(n=len(elegidos), quienes=quienes, voseo=_VOSEO,
                            quien_es=quien_es_texto(ident),
                            tono=TONES.get(tone, TONES[DEFAULT_TONE]),
                            crudeza=crudeness_text(crudeness), edad=edad)
    # system="": el de vlm.py es para el pie de foto (situacional, primera
    # persona) y acá lo pelearía. Los comentarios se guían solos con el prompt.
    raw = vlm.describe(image_path, model=model or vlm.DEFAULT_MODEL,
                       prompt=prompt, num_predict=45 * n, temperature=0.95,
                       system="")
    # además de las personalidades, se le escapan las reglas del prompt
    ecos = [p for _, p in elegidos] + [_VOSEO, quien_es_texto(ident),
                                       crudeness_text(crudeness),
                                       TONES.get(tone, TONES[DEFAULT_TONE])]
    got = _sin_eco(_parse(raw), ecos)
    if not got:
        raise RuntimeError(f"El modelo no devolvió comentarios usables: {raw[:120]}")
    # el i-ésimo comentario es del i-ésimo comentador (se lo pedimos en orden)
    con_firma = [{"user": u, "text": _rioplatense(t)} for (u, _), t in zip(elegidos, got)]
    _put(name, rel, comments=con_firma)
    return con_firma
