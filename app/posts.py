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
import random
import re
import threading
from pathlib import Path

from app import NSFW_DIR_NAME, POSTS_PATH
from app import finder, moldes, person_meta, sexpos, vlm
from app import tags as tags_mod
from app.config import load_config
from app.known import load_known

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
# Etiquetas que HABILITAN lo subido de tono. Sin una de estas no hay crudeza,
# aunque tampoco haya etiqueta de menor (ver identidad()): la duda cierra, no
# abre. Son pocas y a propósito: es una marca que ponés vos a mano, no algo que
# adivine el modelo. Las de edad que sugiere la IA ("veinteañera", "madura") NO
# cuentan — si contaran, el modelo volvería a decidir solo y esto no serviría.
_ADULTO = {"adulto", "adulta", "mayordeedad"}


def _norm_tag(t):
    return re.sub(r"[^\wáéíóúñü]", "", t.lower())


def identidad(tags):
    """De las etiquetas de la persona:
    {"genero": "f"|"m"|None, "menor": bool, "sano": bool}.

    - "genero" para que el texto concuerde ("qué linda" vs "qué lindo").
    - "menor": la persona está etiquetada como chico. Manda el LENGUAJE: se le
      habla como a un chico (ver quien_es_texto).
    - "sano": modo seguro, la COMPUERTA. True salvo que haya etiqueta explícita
      de adulto. Apaga la crudeza, saca a los comentadores que desean y no deja
      correr el detector de actos.

    Ojo con la diferencia, que es el punto de todo esto: sin etiquetas, "menor"
    es False pero "sano" es True. O sea, a una persona sin etiquetar NO se la
    trata como a un nene (sería absurdo), pero TAMPOCO se le escribe nada subido
    de tono. Antes esto fallaba al revés: sin etiquetas se asumía adulto y salía
    todo crudo. Si falla, que falle para el lado seguro."""
    t = {_norm_tag(x) for x in tags}
    genero = "f" if t & _GENERO_F else ("m" if t & _GENERO_M else None)
    menor = bool(t & menor_set())
    # la etiqueta de menor le gana a la de adulto: si están las dos, es menor
    adulto = bool(t & adulto_set()) and not menor
    return {"genero": genero, "menor": menor, "sano": not adulto}


def identidad_foto(name, rel):
    """identidad() pero mirando TAMBIÉN las etiquetas de la foto, no solo las del
    perfil.

    Es lo que hay que usar siempre que se sepa de qué foto se trata. "adulta" o
    "nena" se etiquetan naturalmente en la foto (es donde estás mirando cuando la
    ponés), y si solo miráramos el perfil una foto marcada `adulta` seguiría en
    modo sano —pasó— o, peor, una marcada `nena` no activaría la protección."""
    return identidad(tags_mod.effective_photo(name, rel))


def _tags_para_pie(name, rel):
    """Las etiquetas efectivas de la foto (propias + heredadas de la persona),
    sin las de género/edad (esas ya van aparte, por quien_es_texto/quien_es_pie).

    Es la intención confirmada: el pie y los comentarios se escriben A PARTIR de
    esto, literal, en vez de que el modelo se floree con lo que cree ver."""
    fuera = _GENERO_F | _GENERO_M | menor_set() | adulto_set()
    return [t for t in tags_mod.effective_photo(name, rel) if _norm_tag(t) not in fuera]


def _intencion(name, rel, image_path, sano):
    """El "intento confirmado" que alimenta el pie y los comentarios: las
    etiquetas de la foto MÁS, si es NSFW y la persona está confirmada como
    adulta, el acto que detecta el clasificador dedicado (porntech/sex-position).
    Así el texto sabe qué está pasando aunque no hayas aceptado el acto como
    etiqueta a mano.

    Salvaguarda: solo corre con adulto confirmado (sano=False) y en fotos NSFW.
    Sin etiqueta de adulto no corre, aunque la foto esté en NSFW: la duda cierra.
    Fuera de NSFW tampoco, porque ahí el clasificador tira cualquier cosa."""
    tags = _tags_para_pie(name, rel)
    if not sano and NSFW_DIR_NAME in Path(rel).parts:
        try:
            acto = sexpos.clasificar(image_path)
            if acto and _norm_tag(acto[0]) not in {_norm_tag(t) for t in tags}:
                tags.append(acto[0])
        except Exception:
            pass          # el clasificador no debe romper la generación del texto
    return tags

# Un 4b escribe en español neutro por más que le pidas "rioplatense": no sabe qué
# es. Prohibirle las formas concretas SÍ funciona, y como es una regla de
# gramática y no un ejemplo, no la copia textual (los ejemplos sí los copiaba).
# La misma regla que usa el pie (vlm.VOSEO_REGLA): un solo lugar para el voseo y
# la prohibición de españolismos, así el pie y los comentarios hablan igual.
_VOSEO = vlm.VOSEO_REGLA

# Misma lógica que _VOSEO: "sin eufemismos" no le dice NADA a un 4b, no sabe
# cuáles son. Darle la sustitución concreta sí funciona, y como es una regla de
# vocabulario (no un comentario de ejemplo) no la copia textual. Las palabras son
# las mismas que usa el endurecedor de etiquetas (vlm._CRUDO_MAP), para que el
# pie, los comentarios y las etiquetas hablen igual.
_CRUDO_VOCAB = (
    "Nombrás las cosas con la palabra vulgar, nunca la clínica ni el eufemismo: "
    "\"tetas\" y no \"senos\" ni \"pechos\" ni \"busto\"; \"culo\" y no \"glúteos\" "
    "ni \"nalgas\" ni \"trasero\"; \"concha\" y no \"vagina\" ni \"vulva\" ni "
    "\"entrepierna\"; \"pija\" y no \"pene\" ni \"miembro\"; \"en bolas\" y no "
    "\"desnuda\" ni \"sin ropa\". Prohibido \"figura\", \"silueta\", \"curvas\", "
    "\"anatomía\", \"escultural\" y \"despampanante\": esas palabras no se usan acá.")

# Desde qué nivel de la perilla se le suma la regla de vocabulario (0 sano,
# 1 pícaro no la llevan; 2 subido en adelante sí). Con menores la crudeza se
# fuerza a 0 antes de llegar acá, así que nunca les toca.
_CRUDO_DESDE = 2

# LA VOZ PROVOCADORA (pedido del usuario), y va SOLO en el nivel más alto de la
# perilla (guarro). El pie ya es primera persona del personaje; lo que le faltaba
# era DIRIGIRSE AL QUE MIRA. La diferencia es de a quién le habla, no de crudeza:
#
#   - La VISION (app/vision2.py) es el que MIRA: desea a la de la foto ("le
#     metería un pijazo"). El deseo va de él HACIA ella.
#   - El PIE es la de la foto: le habla al VIEWER y lo provoca ("mirá cómo
#     estoy", "esto es para vos"). Misma crudeza, dirigida al revés.
#
# Sin esta regla, el nivel guarro escribía cosas subidas pero PARA SÍ MISMA
# ("estoy tan mojada"), que no es provocar a nadie. Con esto se le pide el
# aguijón hacia afuera. Es una regla de estilo, no un ejemplo: por eso no se
# copia textual (misma razón que _CRUDO_VOCAB, ver arriba).
#
# Solo en el nivel más alto a propósito (decisión del usuario: "mejorar solo el
# nivel más alto"): los otros cuatro quedan como están.
_PROVOCA_VOCAB = (
    "Le hablás AL QUE ESTÁ MIRANDO la foto, no a vos misma: lo provocás, lo "
    "calentás, lo citás. Nada de describirte desde adentro (\"estoy mojada\", "
    "\"tengo ganas\") como si te lo contaras a vos: eso no provoca a nadie. "
    "Hablale a él: \"mirá lo que tengo\", \"vení a buscarlo\", \"esto es para "
    "vos\", \"¿aguantás verlo sin tocarte?\". Lo que mostrás es una invitación y "
    "el pie se lo dice a él, cara a cara.")

# EL PIE DESDE LA VISION (pedido del usuario). Cuando la foto YA tiene su
# «visión» guardada (paso 1 del Centro de IA, app/vision2.py), el pie se escribe
# A PARTIR DE ESE TEXTO y NO vuelve a mirar la foto.
#
# POR QUÉ: el pie mirando la foto INVENTA. Medido sobre acro_0008 (bragas grises
# puestas, guardadas): el pie viejo dijo "un pantalón gris de lado", "resaltando
# el vello púbico" (no se ve) y —en otra corrida— "acostada sobre un mueble de
# baño", un lugar que no está. La visión de esa MISMA foto, en cambio, es fiel y
# anclada: dice "bragas grises de cintura baja", "top negro", "uñas pintadas de
# rojo", "pared blanca o azulejo claro". Y esa visión ya pasó por el filtro que
# costó todo el día (dos pasadas, hechos anclados, el detector de freno).
#
# Escribiendo desde la visión, el pie HEREDA esa fidelidad: probado 5 corridas x
# 2 fotos, cero inventos de lugar y las prendas correctas ("bragas", no
# "pantalón"). Es el mismo golpe que arregló la narración: quitarle a un modelo
# la foto que mira, para que no rellene lo que no ve.
#
# NO ES la voz del narrador de la visión (que desea A ELLA). Es la voz del
# PERSONAJE: ella le habla al VIEWER y lo provoca (ver _PROVOCA_VOCAB). Misma
# crudeza, dirigida al revés. Por eso el system es propio y no reusa _RED_S.
#
# La REGLA DE ORO acá es la misma de la visión: no agregar NADA que la
# descripción no diga — ni el lugar, ni la ropa, ni el cuerpo. Si la descripción
# dice "bragas", son bragas y nunca "short"; si no nombra un lugar, el pie no
# tiene lugar. El andamiaje va en inglés (los modelos obedecen mejor) pero la
# salida se pide en español desde el propio system: no hay traductor.
_SYS_PIE_VISION = """You write the caption for a photo of YOURSELF, as if you
were posting it on an xxx social network.

YOU ARE the person in the photo, writing in the FIRST PERSON. Never describe
yourself from outside, never in third person: no "in the photo you can see",
"a girl who...", "she is". If you cannot say it with "I am/I have/I feel", it is
wrong.

You do NOT see the photo: you only have a DESCRIPTION of it, and that
description is all you know. Do not add anything that is not in it — not the
place, not the clothes, not the body. If the description says panties, it is
panties, never shorts. If it does not name a place, the caption has no place.

YOU TALK TO WHOEVER IS LOOKING at the photo, not to yourself: you provoke him,
you turn him on, you invite him. Do not describe yourself from inside as if you
were telling it to yourself ("I am wet", "I feel like it"): that does not do
anything to the one watching. What you show is an invitation, and the caption
says it to him, face to face.

Crude and direct: the things by their name, no euphemisms, no beating around
the bush. One or two sentences. Short. One paragraph, never lists, titles or
markdown.

Write the caption in Spanish (rioplatense), never in English."""


# El modelo de texto que redacta el pie desde la visión. Es el mismo redactor de
# la visión (app/vision2.REDACTOR): abliterated, para que no se frene en lo
# subido. Se puede pisar desde Ajustes con la misma clave que la visión
# ("vision_redactor"), para no tener dos modelos que mantener.
_VISION_REDACTOR = "huihui_ai/qwen3.5-abliterated:9b"


def _redactor_vision():
    """El redactor configurado para la visión, o el de fábrica."""
    return load_config().get("vision_redactor") or _VISION_REDACTOR


def _pie_desde_vision(name, rel, vision, voz="", crudeza="", model=None):
    """El pie escrito desde la visión guardada (sin volver a mirar la foto).

    Devuelve "" si no se pudo (el llamador cae al camino viejo, mirando la foto).
    Va rápido: es una sola pasada de texto, sin los tokens de la imagen.
    """
    pr = "DESCRIPTION of the photo:\n\n%s\n\n" % vision
    if voz:
        pr += "How you talk: %s\n\n" % voz
    if crudeza:
        pr += "Crudeness: %s\n\n" % crudeza
    pr += "Write the caption now, one paragraph, in Spanish."
    try:
        txt = vlm.generar_texto(pr, model=model or _redactor_vision(),
                                system=_SYS_PIE_VISION,
                                num_predict=220, temperature=0.75)
    except Exception:
        return ""
    txt = vlm.clean_caption(txt)
    txt = _sin_instrucciones(txt, [pr, _SYS_PIE_VISION])
    return _sin_cola_cortada(_sin_repetidas(txt))


_PROMPT = """Mirá esta foto. Estas {n} personas la están comentando en una red
social. Escribí el comentario de cada una, en su voz:

{quienes}
{pie}{compania}{etiquetas}
Reglas estrictas:
- {quien_es}
- {voseo}
- Cada uno comenta ALGO CONCRETO de lo que está pasando en la foto: la postura,
  lo que está haciendo, cómo está puesta, qué se le ve. Si está sentada de
  piernas abiertas, se comenta ESO y si les gusta o no. Está PROHIBIDO el elogio
  genérico que serviría para cualquier otra foto ("qué linda", "te queda
  increíble", "estás radiante"): eso no dice nada.
- Si en la foto no pasa nada, comentan que no pasa nada. No inventen.
- Que se note quién es cada uno: tienen que sonar DISTINTOS entre sí. Cada uno
  habla de OTRA cosa (uno de la ropa, otro de la pose, otro del lugar, otro tira
  un chiste). PROHIBIDO que dos digan lo mismo con otras palabras: si uno ya dijo
  "qué linda", ningún otro vuelve a hablar de lo linda que está.
- MÁXIMO 10 PALABRAS cada uno. Son comentarios, no párrafos.
- El clima general es: {tono}
- Qué tan subidos: {crudeza}
{edad}- Un comentario por línea, en el mismo orden, y NADA más: sin numerar, sin
  poner quién lo dice, sin comillas, sin encabezado, sin explicar nada.
"""


# Un solo escritor por vez sobre posts.json. Ver _put: el pie y los comentarios
# se generan en paralelo y tardan minutos, así que sin esto se pisan.
_LOCK = threading.RLock()


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


def _post_de(got):
    """Normaliza un post crudo del json. "people" = OTROS conocidos etiquetados
    en la foto (el dueño de la carpeta va implícito). "fav" = favorito.
    "faces" = etiquetado A MANO con posición (ver set_faces): [{"name", "box"}],
    con el box normalizado 0-1 — no en píxeles, así vale para cualquier tamaño
    de pantalla y para cualquier copia de la foto."""
    got = got or {}
    return {"caption": got.get("caption"), "comments": got.get("comments", []),
            "people": got.get("people", []), "fav": bool(got.get("fav")),
            "vision": got.get("vision"), "faces": _faces_limpias(got.get("faces"))}


def _faces_limpias(faces):
    """Sanea la lista de caras etiquetadas a mano. Descarta lo que no sirva en
    vez de romper: un box sin las cuatro coordenadas, o con valores fuera de
    0-1, se tira (mejor perder una etiqueta que pintar un cuadro en el aire)."""
    limpias = []
    for f in faces or ():
        try:
            n = str(f.get("name") or "").strip()
            b = [float(x) for x in f.get("box")]
            if not n or len(b) != 4:
                continue
            if any(x < -0.02 or x > 1.02 for x in b):
                continue
            x0, y0, x1, y1 = [min(1.0, max(0.0, x)) for x in b]
            if x1 - x0 < 0.005 or y1 - y0 < 0.005:
                continue                 # cuadro degenerado: no es una cara
            limpias.append({"name": n, "box": [x0, y0, x1, y1]})
        except (TypeError, ValueError, KeyError):
            continue
    return limpias


def set_fav(name, rel, value):
    """Marca/desmarca la foto como favorita."""
    return bool(_put(name, rel, fav=bool(value))["fav"])


def get(name, rel):
    """El post de una foto: {"caption": str|None, "comments": [...], "people": [...]}.

    Siempre devuelve el dict completo, aunque la foto no tenga nada todavía."""
    return _post_de(_load().get(_key(name, rel)))


def get_many(name, rels):
    """Los posts de varias fotos de una, sin releer el archivo por cada una."""
    data = _load()
    return {rel: _post_de(data.get(_key(name, rel))) for rel in rels}


def set_faces(name, rel, faces):
    """Las caras etiquetadas A MANO con su cuadro: [{"name", "box"}, ...].

    `box` viene en normalizado 0-1 (x0, y0, x1, y1) desde el frontend: la esquina
    sup-izq y la inf-der del cuadro que dibujó el usuario, divididas por el ancho
    y alto de la foto mostrada. Se guarda normalizado y no en píxeles porque los
    píxeles dependen de cómo se mostró la foto; el normalizado vale siempre.

    A diferencia de set_people (que exige que el nombre exista en la base), acá
    el nombre es libre: la idea es poder marcar la cara de la DUEÑA (que no está
    en su propia base de conocidos) y también la de alguien que no esté cargado.
    Sin dueño no hay a quién atribuir la ropa, que es el problema que esto viene
    a resolver."""
    limpio, vistos = [], set()
    for f in faces or ():
        try:
            n = str(f.get("name") or "").strip()
            b = [float(x) for x in f.get("box")]
        except (TypeError, ValueError, KeyError):
            continue
        if not n or len(b) != 4:
            continue
        if n in vistos:
            continue
        vistos.add(n)
        limpio.append({"name": n, "box": b})
    return _put(name, rel, faces=_faces_limpias(limpio))["faces"]


def set_people(name, rel, people):
    """Quiénes MÁS aparecen en la foto (etiquetado a mano con @ en el modal).

    Se guardan solo los que existen en la base y no son el dueño (María no se
    etiqueta a sí misma en su propia foto)."""
    conocidos = set(load_known())
    limpio, vistos = [], set()
    for p in people:
        p = (p or "").strip().lstrip("@")
        if p and p != name and p in conocidos and p not in vistos:
            vistos.add(p)
            limpio.append(p)
    return _put(name, rel, people=limpio)["people"]


def _put(name, rel, **campos):
    """Guarda campos del post SIN pisar lo que haya escrito otro mientras tanto.

    El pie y los comentarios se generan al mismo tiempo si apretás los dos
    botones, y cada uno tarda MINUTOS. Sin el lock (probado): los dos leen el
    archivo al empezar, y el que termina último guarda su copia vieja y le borra
    el trabajo al otro. Por eso se lee de nuevo ACÁ ADENTRO, después de esperar,
    y no antes de llamar al modelo."""
    with _LOCK:
        data = _load()
        post = data.get(_key(name, rel)) or {}
        post.update(campos)
        data[_key(name, rel)] = post
        _save(data)
        return post


def set_caption(name, rel, text):
    """Guarda el pie (también sirve para reescribirlo a mano)."""
    return _put(name, rel, caption=text)["caption"]


def set_vision(name, rel, text):
    """Guarda la «visión» de la foto: lo que la IA ve que hay (paso 1 del
    Centro de IA). Sin pisar el pie ni los comentarios."""
    return _put(name, rel, vision=text)["vision"]


def set_comments(name, rel, comments):
    return _put(name, rel, comments=comments)["comments"]


def clear(name, rel):
    """Borra el post entero (pie + comentarios) de una foto."""
    with _LOCK:
        data = _load()
        if data.pop(_key(name, rel), None) is not None:
            _save(data)


def clear_comments(name, rel):
    """Borra solo los comentarios, para regenerarlos sin perder el pie."""
    with _LOCK:
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
    with _LOCK:
        data = _load()
        got = data.pop(_key(name, old_rel), None)
        if got is not None:
            data[_key(name, new_rel)] = got
            _save(data)


def forget_person(name):
    """Se borró la persona: sus posts se van, y su nombre se saca de las fotos de
    los demás donde estaba etiquetada (si no, queda un @ colgado de un fantasma)."""
    with _LOCK:
        data = _load()
        cambio = False
        for k in [k for k in data if k.startswith(f"{name}/")]:
            data.pop(k); cambio = True
        for post in data.values():
            if name in post.get("people", []):
                post["people"] = [p for p in post["people"] if p != name]
                cambio = True
        if cambio:
            _save(data)


def rename_person(old, new):
    with _LOCK:
        data = _load()
        for k in [k for k in data if k.startswith(f"{old}/")]:
            data[f"{new}/{k.split('/', 1)[1]}"] = data.pop(k)
        # y donde estaba etiquetada en fotos de otros, cambiarle el nombre
        for post in data.values():
            if old in post.get("people", []):
                post["people"] = [new if p == old else p for p in post["people"]]
        _save(data)


# --- generación (lenta: es el modelo de visión) ---------------------------

def crudeness_text(level):
    """El texto del nivel de la perilla, acotado por si la config trae fruta."""
    try:
        i = int(level)
    except (TypeError, ValueError):
        i = CRUDENESS_DEFAULT
    niveles = crudeness_list()      # fábrica + lo que reescribiste en Ajustes
    i = max(0, min(i, len(niveles) - 1))
    txt = niveles[i][1]
    # de "subido" para arriba se le suma QUÉ palabras usar: sin esto el 4b
    # entiende "sin eufemismos" pero igual escribe "senos" y "figura escultural"
    if i >= _CRUDO_DESDE:
        txt += " " + _CRUDO_VOCAB
    # la voz provocadora SOLO en el nivel más alto (guarro = el último): el pie
    # deja de escribirse para sí mismo y le habla al que mira (ver _PROVOCA_VOCAB)
    if i == len(niveles) - 1:
        txt += " " + _PROVOCA_VOCAB
    return txt


def generate_caption(name, rel, image_path, model=None,
                     crudeness=CRUDENESS_DEFAULT, diminutivos=False,
                     usar_moldes=True):
    """Escribe el pie de foto y lo guarda. ~1,5 minutos en CPU.

    Sabemos de quién es la foto (sale de la carpeta), así que el pie lo escribe
    ELLA, en primera persona y con SU voz (la que configuraste en su perfil).

    Quién MÁS está en la foto: si etiquetaste a mano (con @ en el modal) se usa
    eso; si no, se cae al reconocimiento de caras. Así el pie dice "acá comiendo
    con Alberto" y no "con alguien".

    Las ETIQUETAS de la foto (propias + de la persona) son la intención real:
    el pie se escribe A PARTIR de ellas, literal, no de lo que el modelo cree
    ver — evita que se floree o invente un clima que no está."""
    # el pie va en primera persona, así que el género importa: "estoy cansada"
    # o "cansado" sale de las etiquetas de la persona
    ident = identidad_foto(name, rel)
    if ident["sano"]:
        crudeness = 0

    # MOLDES PRIMERO, antes de tocar nada caro. Escritos por el usuario, salen
    # instantáneos y con su voz; el modelo local, probado en seis configuraciones,
    # no llega a eso (ver app/moldes.py). Si ninguna etiqueta prende un molde se
    # cae al modelo, así una foto sin cobertura igual tiene pie.
    # Va ANTES del reconocimiento de caras a propósito: who_is_in carga la base de
    # vectores y escanea la imagen, y no tendría sentido pagar eso para después
    # contestar con un molde que no lo usa.
    if usar_moldes:
        # el pie que ya tiene: se le pasa para que devuelva OTRO. Sin esto,
        # apretar "regenerar" te daba siempre exactamente el mismo.
        # `pulir` = lo mismo que se le hace al pie ANTES de guardarlo. Va también
        # a escribir() para que compare el pie guardado contra las variantes ya
        # transformadas; si no, nunca coincide y regenerar devuelve siempre el
        # primero (pasó con los diminutivos: guardaba "playita", comparaba "playa").
        pulir_pie = con_diminutivos if diminutivos else None
        # "Solo mis pies" (Ajustes): los 62 de fábrica quedan horneados en tu
        # moldes.json al guardar el primero y después compiten con los tuyos, así
        # que el pie variaba mucho antes de caer en uno propio.
        solo_mios = bool(_cfg().get("solo_moldes_propios"))
        pie = moldes.escribir(_key(name, rel),
                              tags_mod.effective_photo(name, rel), sano=ident["sano"],
                              evitar=get(name, rel)["caption"], pulir=pulir_pie,
                              solo_propios=solo_mios)
        if pie:
            if pulir_pie:
                pie = pulir_pie(pie)
            _put(name, rel, caption=pie)
            return pie
        if solo_mios:
            # con "solo mis pies" no se cae al modelo: se avisa, que es lo útil
            # (te dice dónde falta escribir un molde en vez de tapar el agujero)
            raise RuntimeError("No hay un pie tuyo para estas etiquetas. "
                               "Escribí uno con ✍ (o destildá «Solo mis pies» en Ajustes).")

    # EL PIE DESDE LA VISION, si la foto ya la tiene. Va ACÁ: después de los
    # moldes (que son instantáneos y son la voz del usuario) y antes del
    # reconocimiento de caras (que es caro y el camino por visión no lo necesita:
    # las personas ya no entran, el material es la descripción guardada).
    #
    # Solo con crudeza real (crudeness > 0): el system de acá es crudo y
    # provocador, así que un pie sano —menor o sin adulto confirmado, donde la
    # crudeza se fuerza a 0 arriba— NO puede pasar por este camino. Esa compuerta
    # es la misma del pie de siempre y se respeta igual.
    #
    # SI LA VISION FALLA LA VOZ, SE REINTENTA — no se cae al camino que mira la
    # foto. Eso se midió: una corrida del 9B se fue a describir desde afuera
    # ("Estáis viendo la foto de una chica... Están en un baño") y al rechazarla
    # el código caía al camino viejo, que MIRANDO LA FOTO inventó el baño y los
    # "pantalones cortados" — el defecto entero que este camino existe para
    # matar. Mejor reintentar la voz que volver al que inventa: el material
    # anclado es la razón de ser de todo esto.
    if crudeness > 0:
        vision = (get(name, rel).get("vision") or "").strip()
        if vision:
            voz = person_meta.get_voice(name)
            crudo = crudeness_text(crudeness)
            for _ in range(3):            # 3 tiros: el 9B yerra la voz de vez en cuando
                texto = _pie_desde_vision(name, rel, vision, voz=voz, crudeza=crudo)
                texto = _rioplatense(texto) if texto else texto
                if texto and _sirve_pie_vision(texto):
                    if diminutivos:
                        texto = con_diminutivos(texto)
                    _put(name, rel, caption=texto)
                    return texto
            # 3 intentos y ninguno sirvió: se avisa en vez de inventar mirando
            raise RuntimeError(
                "El pie desde la visión no salió bien (el modelo describió desde "
                "afuera). Probá de nuevo, o regenerá la visión de esta foto.")

    con = get(name, rel)["people"]      # etiquetados a mano, si hay
    if not con:
        try:
            con = finder.who_is_in(image_path, exclude=[name])
        except Exception:
            con = []      # sin base de caras o foto ilegible: el pie igual sale
    voz = person_meta.get_voice(name)
    crudeza, quien_es = crudeness_text(crudeness), quien_es_pie(ident)
    prompt = vlm.caption_prompt(name, voz, con=con, crudeza=crudeza, quien_es=quien_es,
                                etiquetas=_intencion(name, rel, image_path, ident["sano"]),
                                system=_cfg().get("post_pie_system") or "narrador")
    extra = (_cfg().get("post_extra_pie") or "").strip()   # tu caja de Ajustes
    if extra:
        prompt += "\n" + extra
    # el arranque rota por foto (estable): así no salen todos con "Acá..."
    ops = vlm.PREFILL_OPCIONES
    arranque = ops[_stable_hash(_key(name, rel)) % len(ops)]
    modelo = model or vlm.DEFAULT_MODEL

    def escribir(prefill):
        # El modelo repite lo que le pusiste en el prompt: real, devolvió "Acá en
        # la playa, ser directo, vulgar..." y pies terminados en "Sos Alejandro".
        # Se compara contra el prompt ENTERO, no contra los pedazos: cualquier
        # línea que le mandemos puede volver copiada.
        # num_predict corto a propósito: el pie son una o dos frases. Con 280 el
        # modelo se queda sin qué decir a la mitad y RELLENA repitiendo el mismo
        # verso hasta gastar el presupuesto (visto: 3 veces la misma frase). Menos
        # techo = menos relleno, y encima tarda menos.
        crudo = vlm.describe(image_path, model=modelo, prompt=prompt, prefill=prefill,
                             num_predict=100)
        # _rioplatense corrige el tuteo y lo peninsular ("chaval"->"pibe",
        # "polla"->"pija"). Estaba SOLO en los comentarios, así que al pie le
        # llegaban gallegismos tal cual: real, devolvió "un chaval de 25 años".
        return crudo, _rioplatense(_sin_repetidas(_sin_autonombre(
            _sin_instrucciones(vlm.clean_caption(crudo), [prompt, vlm.SYSTEM_PIE]),
            name)))

    # SIN arranque primero: es el camino rápido (medido con el 7B de captions:
    # 30s sin arranque contra 121s con) y el único que no puede romperle la
    # plantilla del chat a nadie. El arranque solo entra si hace falta.
    if _NECESITA_ARRANQUE.get(modelo):
        raw, text = escribir(arranque)
        if not text or _fuga_de_rol(raw):     # dejó de andar: se vuelve a probar
            _NECESITA_ARRANQUE[modelo] = False
            raw, text = escribir(None)
    else:
        raw, text = escribir(None)
        # sin arranque algunos describen desde afuera ("una chica sentada en..."):
        # a esos, y solo a esos, se les pone el arranque y se recuerda.
        if text and not _fuga_de_rol(raw) and not _sirve(text):
            crudo2, text2 = escribir(arranque)
            # el arranque solo se adopta si el resultado sirve DE VERDAD: en
            # primera persona descontando el arranque, y sin tratarla de "vos"
            if text2 and not _fuga_de_rol(crudo2) and _sirve(text2, arranque):
                _NECESITA_ARRANQUE[modelo] = True
                raw, text = crudo2, text2
        elif not text:
            raw, text = escribir(arranque)
    if not text:
        raise RuntimeError("El modelo solo repitió las instrucciones. Probá de nuevo.")
    text = _sin_cola_cortada(text)
    if diminutivos:
        text = con_diminutivos(text)
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
    # Españolismos: al 4b no le alcanza con prohibírselos en el prompt (igual se
    # le escapan), así que se CORRIGEN acá, después de generar. Solo los que son
    # inequívocamente de España: "joder" y "no jodas" acá se usan, no se tocan.
    (r"\bme mola\b", "me encanta"), (r"\bte mola\b", "te gusta"),
    (r"\bmolan\b", "están buenísimas"), (r"\bmola\b", "está buenísimo"),
    (r"\bguay\b", "genial"), (r"\bcoño\b", "la concha"),
    (r"\bfollando\b", "cogiendo"), (r"\bfollar\b", "coger"),
    (r"\bpolla\b", "pija"), (r"\bgilipollas\b", "pelotudo"),
    (r"\bchaval\b", "pibe"), (r"\bchavala\b", "piba"),
    (r"\bhostia\b", "la puta madre"), (r"\bflipo\b", "no lo puedo creer"),
    (r"\bflipar\b", "alucinar"), (r"\bcurrar\b", "laburar"),
    (r"\bmajo\b", "copado"), (r"\bmaja\b", "copada"),
    (r"\bvosotros\b", "ustedes"), (r"\bvale\?", "dale?"),
]


def _rioplatense(text):
    """Le saca el tuteo que se le escapa al modelo. Conserva las mayúsculas."""
    for patron, cambio in tuteo_pairs():   # fábrica + los pares de Ajustes
        def _rep(m, c=cambio):
            return c.capitalize() if m.group(0)[:1].isupper() else c
        text = re.sub(patron, _rep, text, flags=re.IGNORECASE)
    return text


# Diminutivos, el habla de acá: "acá con friito", "comiendo una sopita".
#
# Es una LISTA y no un motor de reglas, a propósito. El diminutivo español es
# irregular (frío->friito, despacio->despacito, cerveza->cervecita,
# flaco->flaquito) y sin saber qué palabra es un verbo, una regla te deja
# "tomandito". Con lista: poca cobertura, cero engendros. Agregá las que uses.
#
# Tampoco se lo pide al modelo: probado, lo ignora entero (cero diminutivos,
# pidiéndoselo con ejemplos y todo).
#
# Dos formas, las dos válidas:
#   "sopa": "sopita"                          -> siempre esa
#   "tranquilo": ["tranquilito", "tranquilazo"]  -> una al azar cada vez
DIMINUTIVOS = {
    # temperatura y estados
    "frío": "friito", "frio": "friito", "fría": "friita", "fria": "friita",
    "calor": "calorcito", "caliente": "calentito", "solo": "solito",
    "sola": "solita", "cansado": "cansadito", "cansada": "cansadita",
    "borracho": "borrachito", "borracha": "borrachita",
    # cuerpo y ropa
    "desnudo": "desnudito", "desnuda": "desnudita", "gordo": "gordito",
    "gorda": "gordita", "flaco": "flaquito", "flaca": "flaquita",
    "ropa": "ropita", "vestido": "vestidito", "bikini": "bikinito",
    "pollera": "pollerita", "remera": "remerita", "short": "shorcito",
    "tanga": "tanguita", "corpiño": "corpiñito", "zapatos": "zapatitos",
    # comida y bebida
    "sopa": "sopita", "torta": "tortita", "helado": "heladito",
    "cerveza": "cervecita", "vino": "vinito", "mate": "matecito",
    "café": "cafecito", "cafe": "cafecito", "agua": "agüita", "pan": "pancito",
    "asado": "asadito", "picada": "picadita",
    # lugares y cosas
    "casa": "casita", "cama": "camita", "playa": "playita", "sol": "solcito",
    "auto": "autito", "perro": "perrito", "gato": "gatito", "foto": "fotito",
    "fiesta": "fiestita", "baño": "bañito", "cocina": "cocinita",
    "jardín": "jardincito", "patio": "patiecito", "pieza": "piecita",
    # modos
    "despacio": "despacito", "callado": "calladito", "callada": "calladita",
    "quieto": "quietito", "quieta": "quietita", "juntos": "juntitos",
    "juntas": "juntitas", "cerca": "cerquita", "poco": "poquito",
    "chico": "chiquito", "chica": "chiquita", "grande": "grandote",
    "lindo": "lindito", "linda": "lindita", "feo": "feíto", "fea": "feíta",
    "nuevo": "nuevito", "nueva": "nuevita", "viejo": "viejito",
    "vieja": "viejita", "rápido": "rapidito",
    # ejemplo de la forma con varias: sale una al azar
    "tranquilo": ["tranquilito", "tranquilazo"],
    "tranquila": ["tranquilita", "tranquilaza"],
}
# Piropos: van SOLO en los comentarios, no en el pie. Un piropo se le dice A
# alguien, y el pie lo escribe la persona sobre sí misma ("acá con friito"): ahí
# un "mamita" no tiene sentido.
#
# Y van SOLO en fotos de adultos: para un chico se saltean enteros (ver
# con_diminutivos). Los diminutivos comunes sí valen para todos — a un chico le
# quedan mejor que a nadie.
PIROPOS = {
    "mamá": ["mamita", "mami"], "mama": ["mamita", "mami"],
    "papá": ["papito", "papi"], "papa": ["papito", "papi"],
    "reina": ["reinita", "reinaza"], "diosa": ["diosita", "diosaza"],
    "muñeca": ["muñequita"], "princesa": ["princesita"],
    "bombón": ["bomboncito"], "bombon": ["bomboncito"],
    "hermosa": ["hermosita"], "hermoso": ["hermosito"],
    "potra": ["potrita"], "yegua": ["yeguita"], "mina": ["minita"],
    "tesoro": ["tesorito"], "amor": ["amorcito"], "cielo": ["cielito"],
    "corazón": ["corazoncito"], "corazon": ["corazoncito"],
    "negro": ["negrito"], "negra": ["negrita"],
    "rubio": ["rubito"], "rubia": ["rubita"],
    "morocho": ["morochito"], "morocha": ["morochita"],
}


def _armar_re(tabla):
    return re.compile(r"\b(%s)\b" % "|".join(sorted(tabla, key=len, reverse=True)), re.I)


# --- overrides de Configuración -------------------------------------------
# Todo lo de acá se puede tocar desde Ajustes. La idea: lo del código es la BASE
# y lo de Ajustes se SUMA (no reemplaza), así nadie desactiva sin querer una
# palabra que protege (menores, sexuales). Vacío en Ajustes = solo la base.


def _cfg():
    try:
        c = load_config()
        # El pie en "narrador" adivina todo ("una chica que..."), en 
        # "explicito" te pone "tetas al aire", "culito roto"...
        c.setdefault("post_pie_system", "narrador")
        return c
    except Exception:
        return {}


def _nw(w):
    return re.sub(r"[^\wáéíóúñü]", "", (w or "").strip().lower())


def menor_set():
    """Etiquetas que marcan menor: las de fábrica + las que sumaste en Ajustes."""
    return _MENOR | {_nw(w) for w in _cfg().get("f_menor_extra", []) if w.strip()}


def adulto_set():
    """Etiquetas que habilitan lo subido de tono.

    A diferencia de menor_set(), esta NO se puede ampliar desde Ajustes, y es a
    propósito: sumar palabras acá AFLOJA la protección en vez de reforzarla, y
    alcanzaría con una palabra común de más ("joven") para volver al viejo
    default de asumir adulto sin que te des cuenta."""
    return set(_ADULTO)


def vacias_set():
    return _VACIAS | {_nw(w) for w in _cfg().get("f_vacias_extra", []) if w.strip()}


def tuteo_pairs():
    """Correcciones tú->vos: las de fábrica + las tuyas ("tienes=tenés")."""
    pares = list(_TUTEO)
    for linea in _cfg().get("f_tuteo_extra", []):
        if "=" in linea:
            a, b = linea.split("=", 1)
            a, b = a.strip(), b.strip()
            if a and b:
                pares.append((rf"\b{re.escape(a)}\b", b))
    return pares


def crudeness_list():
    """Los 5 niveles. Un texto vacío en Ajustes = se usa el de fábrica."""
    ov = _cfg().get("f_crudeness") or []
    out = []
    for i, (clave, txt) in enumerate(CRUDENESS):
        propio = ov[i].strip() if i < len(ov) and isinstance(ov[i], str) and ov[i].strip() else ""
        out.append((clave, propio or txt))
    return out


def _tabla_dim(base, clave):
    """La tabla de diminutivos/piropos: la de fábrica + lo que sumaste en Ajustes.
    El override es {palabra: [forma, ...]} y pisa esa palabra o agrega nuevas."""
    tabla = dict(base)
    extra = _cfg().get(clave) or {}
    if isinstance(extra, dict):
        for k, v in extra.items():
            if k and v:
                tabla[_nw(k)] = v
    return tabla


def diminutivos_map():
    return _tabla_dim(DIMINUTIVOS, "f_diminutivos_extra")


def piropos_map():
    return _tabla_dim(PIROPOS, "f_piropos_extra")


def con_diminutivos(texto, piropos=False):
    """Pasa a diminutivo las palabras conocidas. Respeta las mayúsculas.

    Si una palabra tiene varias formas (una lista), se elige una al azar: así no
    sale siempre la misma en todas las fotos.

    piropos=True suma la tabla PIROPOS (mamá->mamita). Va solo en los comentarios
    y solo si la persona NO está etiquetada como menor.

    Las tablas salen de Configuración (fábrica + lo tuyo), por eso el regex se
    arma acá y no una vez al importar."""
    tabla = {**diminutivos_map(), **piropos_map()} if piropos else diminutivos_map()
    if not tabla:
        return texto
    def cambio(m):
        d = tabla[m.group(0).lower()]
        if not isinstance(d, str):        # una lista: elegimos una
            d = random.choice(list(d))
        return d.capitalize() if m.group(0)[:1].isupper() else d
    return _armar_re(tabla).sub(cambio, texto)


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
    una coincidencia real no.

    Las frases más cortas que n no tienen tirada que comparar (así se colaba "Sos
    Alejandro." al final del pie), así que esas se buscan enteras dentro del
    prompt. Con 2 palabras o más: una sola no dice nada."""
    p = _palabras(texto)
    if len(p) < 2:
        return False
    if len(p) < n:
        corto = " ".join(p)
        return any(corto in " ".join(_palabras(f)) for f in fragmentos)
    mios = set()
    for f in fragmentos:
        mios |= _ngramas(f, n)
    return bool(mios and _ngramas(texto, n) & mios)


# Marcadores de rol del chat largados como texto plano. No todos los modelos
# aceptan el "arranque" (que les pongamos palabras en la boca): a los que no les
# calza la plantilla se les escapa el rol y siguen escribiendo instrucciones en
# vez de un pie. "user"/"assistant" no son palabras en español, así que buscarlas
# sueltas no da falsos positivos en un pie real.
_FUGA_ROL = re.compile(r"\b(?:user|assistant|system)\b|<\|im_(?:start|end)\|>", re.I)
# La fuga no siempre viene como palabra suelta: se vio "Me systema es un chaval
# de 25 años…", con el marcador PEGADO. Buscar "sistema" en cualquier lado daría
# falsos positivos (es una palabra normal), así que solo cuenta si aparece al
# principio, que es donde arranca una fuga de plantilla.
_FUGA_ARRANQUE = re.compile(r"^.{0,20}\bs[yi]stema?\b", re.I)


def _fuga_de_rol(raw):
    """True si al modelo se le escapó un marcador de rol: señal de que el arranque
    le rompió la plantilla y lo que devolvió no es un pie."""
    raw = (raw or "").lstrip()
    return bool(_FUGA_ROL.search(raw) or _FUGA_ARRANQUE.search(raw))


# Marcas de que el pie está en primera persona. Si no hay ninguna, el modelo se
# puso a describir desde afuera ("una chica sentada en...") y ahí sí hace falta
# el arranque para forzarlo.
_PRIMERA_PERSONA = re.compile(
    r"\b(?:estoy|estaba|tengo|tenía|ando|andaba|soy|era|salgo|salí|vengo|vine|"
    r"me|mi|mis|conmigo|acá estoy)\b", re.I)

# Marcas de que el pie se escribió DESDE AFUERA, hablando de ella en tercera
# persona ("en la foto se ve", "una chica que", "ella está"). Es el defecto que
# el pie intenta evitar con el arranque; acá sirve para rechazar el pie por
# visión que se cayó a describir. "ella" solo cuenta como tercera si NO está en
# una frase donde ya hay primera persona... pero separar frases para eso es
# frágil: se piden los marcadores inequívocos y se deja "ella" afuera, que
# también puede ser "ella y yo".
_TERCERA_PERSONA = re.compile(
    r"(?:\ben la foto\b|\ben la imagen\b|\bse (?:ve|observa|aprecia)\b|"
    r"\buna (?:chica|mujer|piba|mina) que\b|\bsu (?:cuerpo|figura)\b)", re.I)


def _en_primera_persona(texto, arranque=""):
    """El pie habla en primera persona.

    OJO con el `arranque`: se lo inyectamos nosotros, así que si no lo sacamos
    antes, buscar "ando" en un texto que EMPIEZA con "Ando" da siempre que sí y
    la verificación se aprueba sola. Pasó: aceptó "Ando osome mirando al
    espejo..." como bueno y encima memorizó que ese modelo servía."""
    t = (texto or "").strip()
    a = (arranque or "").strip()
    if a and t.lower().startswith(a.lower()):
        t = t[len(a):]
    return bool(_PRIMERA_PERSONA.search(t))


# Le habla a OTRO en vez de escribir como ella: "Sos flaca de costado y tu pelo
# castaño...", "Te has tatuado...". Un pie donde la persona se trata de "vos" a
# sí misma está roto, venga del arranque o no.
_SEGUNDA_PERSONA = re.compile(
    r"\b(?:sos|estás|estabas|tenés|tenías|te\s+has|te\s+ves|tu|tus|vos|"
    r"tuyo|tuya|contigo|tienes|eres)\b", re.I)


def _habla_en_segunda(texto):
    return bool(_SEGUNDA_PERSONA.search(texto or ""))


def _sirve(texto, arranque=""):
    """El pie está en primera persona Y no le habla a otro. Las dos cosas: el que
    motivó esto ("Ando osome... Sos flaca de costado y tu pelo castaño...")
    arrancaba en primera y a la segunda frase ya la estaba describiendo a ella."""
    return _en_primera_persona(texto, arranque) and not _habla_en_segunda(texto)


def _sirve_pie_vision(texto):
    """El pie escrito DESDE LA VISIÓN está en primera persona y no describe a
    nadie desde afuera.

    OJO: a diferencia de _sirve, acá "vos" NO es un error. La voz que se pide es
    la del personaje provocando AL VIEWER ("mirá lo que tengo", "¿aguantás
    verlo?"), y eso se dice con "vos" a propósito. El _habla_en_segunda de
    arriba lo tomaría como falla —fue pensado para el pie que miraba la foto y se
    trataba de "vos" a sí misma—, así que acá NO se usa.

    Lo que sí se rechaza es el defecto de siempre: el pie escrito desde afuera
    ("en la foto se ve", "una chica que...", "ella está"). Sin eso el modelo se
    cae a describir la imagen y deja de ser el pie.

    El ARRANQUE no puede ser una descripción de ella en tercera persona. Se midió
    una corrida que empezaba "Está en bolas con esas tetas..." y después seguía
    con "te muevo el culo": entregaba el 80% del pie desde afuera y pasaba el
    filtro igual, porque buscaba la primera persona en CUALQUIER lado del texto.

    OJO: NO se exige primera persona en la primera frase. Eso rechazaba pies
    legítimos que abren hablándole al viewer ("¿Aguantás verme así?", "Vení a
    buscarlo", "Mirá esas tetas..."), que es justo la voz que se pide. Lo que se
    rechaza es que la primera frase esté escrita sobre ella DESDE AFUERA, o sea
    una tercera persona sin ninguna marca de primera persona ni de trato al
    viewer."""
    t = (texto or "").strip()
    if len(t.split()) < 4:
        return False                      # vacío o un retazo: no sirve
    if _abre_describiendo(t):
        return False                      # abre describiéndola desde afuera
    return not _TERCERA_PERSONA.search(t)


# Verbos típicos con que el modelo la describe desde AFUERA cuando se le cae la
# voz de personaje ("Está en bolas", "Se toca el pelo", "Tiene las tetas al
# aire"). En tercera persona del singular, que es como abre una descripción.
_TERCERA_VERBO = re.compile(
    r"^\W*(?:está|esta|tiene|se\s+(?:toca|mira|sienta|queda|pone|acuesta)|"
    r"lleva|luce|muestra|posa|sonríe|sonrie)\b", re.I)


def _abre_describiendo(texto):
    """True si la primera frase describe a la de la foto desde afuera.

    Con una salvedad que importa: si la frase le habla al viewer, es una apertura
    legítima aunque el verbo suene a tercera ("Mirá cómo está..."). Por eso el
    trato al viewer gana."""
    f = _primera_frase(texto)
    if _habla_al_viewer(f):
        return False
    return bool(_TERCERA_VERBO.search(f))


# Marcas de que la frase se dirige AL QUE MIRA (imperativo o pregunta directa).
# No es _SEGUNDA_PERSONA: acá no importa el "vos", importa que lo cite a él. Por
# eso NO alcanza con un "te" suelto — "te muevo el culo" es una afirmación, no
# una invitación, y dejaba pasar aperturas descriptivas ("Está en bolas..., y
# ahora te muevo el culo"). Solo cuentan imperativos y preguntas.
_VIEWER = re.compile(
    r"\b(?:mirá|mira|vení|ven|fijate|fijáte|dale|tocá|tocate|probá|proba|"
    r"aguantás|querés|imaginás|sentís)\b|¿", re.I)


def _habla_al_viewer(texto):
    return bool(_VIEWER.search(texto or ""))


def _primera_frase(texto):
    """La primera frase, para mirar dónde arranca la voz."""
    return re.split(r"(?<=[.!?…])\s+|\n+", (texto or "").strip(), maxsplit=1)[0]


def _sin_cola_cortada(texto):
    """Saca la última frase si quedó a medio terminar.

    Con el techo de tokens el modelo se corta donde sea ("...estás usando tu
    teléfono para hacer"). Mejor un pie más corto que uno colgado."""
    t = (texto or "").strip()
    if not t or t[-1] in ".!?…":
        return t
    corte = max(t.rfind("."), t.rfind("!"), t.rfind("?"))
    return t[:corte + 1].strip() if corte > 0 else t


# Qué modelos necesitan el arranque, aprendido en caliente (modelo -> bool). El
# arranque le fuerza la primera persona al 4B, pero a otros les rompe la
# plantilla del chat: devuelven basura ("Me xxxie la cara de la camera") o largan
# el rol como texto. Como no se puede saber de antemano, se prueba una vez por
# modelo y se recuerda mientras corra la app.
_NECESITA_ARRANQUE = {}


def _sin_repetidas(texto):
    """Saca las frases repetidas del pie, conservando la primera de cada una.

    Un modelo que se queda sin qué decir rellena repitiendo hasta gastar el
    presupuesto de tokens. Bajar num_predict y subir repeat_penalty lo hace mucho
    menos probable, pero no imposible: esto lo garantiza. Se compara por palabras
    (sin puntuación ni mayúsculas) para que "Estoy en la playa." y "estoy en la
    playa!" cuenten como la misma."""
    frases = [f.strip() for f in re.split(r"(?<=[.!?])\s+|\n+", texto or "") if f.strip()]
    vistas, quedan = set(), []
    for f in frases:
        k = " ".join(_palabras(f))
        if k and k not in vistas:
            vistas.add(k)
            quedan.append(f)
    return " ".join(quedan).strip()


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
    """La regla de cómo referirse a la persona, para el prompt.

    Tres casos, no dos: chico confirmado, adulto confirmado, y el del medio (sin
    etiquetar). Ese último va sano pero SIN lenguaje de chico: tratar de "lindo
    chiquito" a una persona que solo no tiene etiquetas sería absurdo."""
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
        quien = ("La persona de la foto es una MUJER: referite a ella en femenino "
                 "(\"qué linda\", \"estás hermosa\"), nunca en masculino.")
    elif ident["genero"] == "m":
        quien = ("La persona de la foto es un HOMBRE: referite a él en masculino "
                 "(\"qué lindo\", \"estás hecho un toro\"), nunca en femenino.")
    else:
        quien = ("No sabemos si es hombre o mujer: no arriesgues el género, escribí "
                 "los comentarios de forma que sirvan para cualquiera.")
    if ident["sano"]:
        # no está confirmada como adulta: comentarios de amigo, sin levante
        quien += (" No sabemos su edad, así que los comentarios son sanos y "
                  "amistosos —de lo que está haciendo, del lugar, de la ropa—. "
                  "NADA sexual, NADA de levante, NADA sobre su cuerpo: está "
                  "terminantemente prohibido acá.")
    return quien


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
        quien = "Sos mujer: escribí en femenino (\"estoy cansada\", \"tirada\")."
    elif ident["genero"] == "m":
        quien = "Sos hombre: escribí en masculino (\"estoy cansado\", \"tirado\")."
    else:
        quien = ""
    if ident["sano"]:
        # sin adulto confirmado: el pie cuenta qué estás haciendo y nada más
        quien += (" El pie es sano: contás qué estás haciendo, dónde estás o "
                  "cómo la estás pasando. NADA sexual ni subido de tono, y nada "
                  "sobre tu cuerpo: está terminantemente prohibido acá.")
    return quien.strip()


def caras_para_prompt(name, rel):
    """Las caras etiquetadas a mano, en texto, para meter en el prompt.

    Es la que resuelve el problema que originó todo esto: la Visión describió la
    ROPA de alguien que estaba atrás, porque el VLM mira la foto entera y no sabe
    cuál de las personas es la dueña. Con las caras marcadas sabe quién está en
    cada cuadro y a quién describir.

    Va en texto y no como ejemplo: ya está medido en este archivo que los
    ejemplos se copian y las etiquetas se recitan. Se dice quién está y que el
    cuerpo que importa es el suyo — sin describir el cuadro con coordenadas, que
    el modelo convertiría en "en el centro de la imagen, a la izquierda".

    Devuelve "" si la foto no tiene caras marcadas (el caso de siempre)."""
    try:
        caras = get(name, rel).get("faces") or []
    except Exception:
        return ""
    if not caras:
        return ""
    quien = []
    for f in caras:
        n = str(f.get("name") or "").strip()
        if n and n not in quien:
            quien.append(n)
    if not quien:
        return ""
    due = name in quien
    if len(quien) == 1 and due:
        return ("Caras marcadas a mano en la foto: solo %s. Es ella y no hay "
                "otra persona: todo lo que se ve es suyo." % name)
    otros = [q for q in quien if q != name]
    txt = "Caras marcadas a mano en la foto: %s." % ", ".join(quien)
    if due:
        txt += (" La protagonista es %s: el cuerpo y la ropa que describís son "
                "LOS SUYOS." % name)
        if otros:
            txt += (" También están %s, pero NO son la protagonista: no le "
                    "atribuyas a %s la ropa ni el cuerpo de %s."
                    % (", ".join(otros), name, " ni de ".join(otros)))
    else:
        # Ninguna cara marcada es la dueña: puede pasar si etiquetaste solo a
        # otra gente. Se avisa para que no la dé por presente ni le invente ropa.
        txt += " Ninguna es la dueña de la carpeta: no la des por presente."
    return txt


def _sin_copia_del_pie(textos, pie):
    """Tira los comentarios que son el pie copiado y pegado.

    Como el pie viaja en el prompt, el modelo lo devuelve tal cual haciéndolo
    pasar por comentario. Pero CITARLO está bien —es el punto de que lo vea—, así
    que solo se tira lo que es copia casi entera, no lo que comparte unas
    palabras."""
    if not pie:
        return textos
    p = " ".join(_palabras(pie))
    return [t for t in textos if " ".join(_palabras(t)) not in p]


# Palabras tan comunes que dos comentarios distintos las comparten igual. No
# cuentan para decidir si dos se parecen, si no "sos linda vos" y "sos genia vos"
# darían casi iguales por el "sos" y el "vos".
_VACIAS = {"que", "qué", "de", "la", "el", "los", "las", "un", "una", "vos",
           "sos", "te", "me", "se", "y", "en", "con", "por", "para", "es",
           "esa", "ese", "esta", "este", "tan", "muy", "más", "mas", "lo",
           "tu", "tus", "ese", "así", "asi", "a", "no", "si", "sí", "mi"}


def _sin_repetidos(textos, previos=()):
    """Tira los comentarios que dicen casi lo mismo que otro.

    El modelo escribe seis y le salen tres calcados ("qué linda estás", "estás
    hermosa hoy", "qué hermosa sos"). Se comparan las palabras con contenido
    (sin las vacías tipo "que", "sos", "vos"): si dos comparten demasiadas, se
    tira el segundo. `previos` son los que ya estaban en la foto, para que "más
    comentarios" traiga cosas nuevas y no repita lo de antes."""
    vac = vacias_set()                 # fábrica + las que sumaste en Ajustes
    def contenido(t):
        return {w for w in _palabras(t) if w not in vac and len(w) > 2}

    vistos = [contenido(p) for p in previos]
    quedan = []
    for t in textos:
        c = contenido(t)
        if not c:                       # todo palabras vacías: no aporta
            continue
        # coeficiente de solapamiento contra cada uno de los ya aceptados
        repetido = any(v and len(c & v) / min(len(c), len(v)) >= 0.55
                       for v in vistos)
        if repetido:
            continue
        vistos.append(c)
        quedan.append(t)
    return quedan


def _regla_compania(people):
    """Le dice al modelo quién MÁS está en la foto, para que alguno lo nombre.

    Son los conocidos que etiquetaste con @ en el modal. Sin nadie, no dice nada.
    """
    if not people:
        return ""
    quienes = " y ".join(people) if len(people) < 3 else \
        ", ".join(people[:-1]) + f" y {people[-1]}"
    return (f"\nEn la foto también está {quienes}. Alguno de los comentarios "
            f"(no todos) puede nombrarlo, como si lo conociera.\n")


def _regla_etiquetas(tags):
    """Lo que hay confirmado en la foto, para que los comentarios sean sobre eso
    y no sobre lo que el modelo cree ver ("campera" en vez de "blusa")."""
    if not tags:
        return ""
    return (f"\nConfirmado que en la foto hay: {', '.join(tags)}. Los comentarios "
            "hablan de ESO, no de otra cosa que se les ocurra.\n")


def _regla_pie(pie, n):
    """Le cuenta al modelo qué dice el pie y quiénes le responden.

    A quiénes les toca se sortea: si siempre respondieran los mismos —o si todos
    arrancaran por el pie— se nota el molde. Entre uno y la mitad, al azar.

    Sin pie no hay sorteo ni línea: comentan la foto y listo."""
    if not pie or n < 1:
        return ""
    cuantos = random.randint(1, max(1, n // 2))
    quienes = sorted(random.sample(range(1, n + 1), cuantos))
    lista = " y ".join(str(i) for i in quienes) if len(quienes) < 3 else \
        ", ".join(str(i) for i in quienes[:-1]) + f" y {quienes[-1]}"
    return (f'\nEl pie que escribió es: "{pie}"\n'
            f"El comentario {lista} le contesta a lo que dice el pie. Los demás "
            f"ni lo miran: comentan lo que se ve en la foto.\n")


def _pick(name, rel, n, users=None, solo_sanos=False, evitar=()):
    """Qué comentadores comentan ESTA foto: n de los COMMENTERS.

    Estable por foto (ver _stable_hash) pero distinto entre fotos, así no
    comentan siempre los mismos seis. Si en Ajustes cambiaste los nombres, se
    usan los tuyos y las personalidades se reparten entre ellos.

    solo_sanos saca a los que desean: es para las fotos de chicos.

    evitar son los que YA comentaron esta foto: al pedir más comentarios entran
    los que faltan, no los mismos otra vez. Si ya comentaron todos, se repiten
    (mejor repetir que no traer nada)."""
    pool = [(u, p) for u, p, desea in COMMENTERS if not (solo_sanos and desea)]
    if users:
        pool = [(users[i % len(users)], p) for i, (_, p) in enumerate(pool)]
    if not pool:
        return []
    libres = [c for c in pool if c[0] not in set(evitar)] or pool
    arranque = _stable_hash(_key(name, rel)) % len(libres)
    return [libres[(arranque + i) % len(libres)] for i in range(min(n, len(libres)))]


def generate_comments(name, rel, image_path, tone=DEFAULT_TONE, model=None,
                      n=N_COMMENTS, users=None, crudeness=CRUDENESS_DEFAULT,
                      sumar=False, diminutivos=False):
    """Escribe los comentarios y los guarda. ~1 minuto en CPU.

    Comenta un puñado de los COMMENTERS, elegidos según la foto: cada uno tiene
    su personalidad y su @usuario, así los seis suenan distinto entre sí.

    sumar=True agrega a los que ya están, como una publicación de verdad: entran
    los comentadores que todavía no habían comentado esta foto.

    Lanza RuntimeError con mensaje legible si Ollama no está, el modelo no existe
    o lo que devolvió no sirve."""
    # Cómo referirse a la persona, según SUS etiquetas (las que ponés de a
    # muchas en la galería). Si dicen que es un chico, la perilla de crudeza no
    # corre y los que desean no comentan: no es negociable desde Ajustes.
    ident = identidad_foto(name, rel)
    if ident["sano"]:
        crudeness = 0
    post = get(name, rel)
    pie = post["caption"]          # los comentarios lo ven: unos le contestan
    ya_estan = post["comments"] if sumar else []
    elegidos = _pick(name, rel, n, users, solo_sanos=ident["sano"],
                     evitar=[c["user"] for c in ya_estan])
    if not elegidos:
        raise RuntimeError("No hay comentadores disponibles (ver COMMENTERS).")
    quienes = "\n".join(f"{i + 1}. {p}" for i, (_, p) in enumerate(elegidos))
    # el de la edad, 1 de cada AGE_EVERY fotos: estable por foto, así regenerar
    # no lo hace aparecer y desaparecer
    edad = _AGE_RULE + "\n" if _stable_hash(_key(name, rel)) % AGE_EVERY == 0 else ""
    prompt = _PROMPT.format(n=len(elegidos), quienes=quienes, voseo=_VOSEO,
                            pie=_regla_pie(pie, len(elegidos)),
                            compania=_regla_compania(post["people"]),
                            etiquetas=_regla_etiquetas(_intencion(name, rel, image_path, ident["sano"])),
                            quien_es=quien_es_texto(ident),
                            tono=TONES.get(tone, TONES[DEFAULT_TONE]),
                            crudeza=crudeness_text(crudeness), edad=edad)
    extra = (_cfg().get("post_extra_com") or "").strip()   # tu caja de Ajustes
    if extra:
        prompt += "\n- " + extra
    # system="": el de vlm.py es para el pie de foto (situacional, primera
    # persona) y acá lo pelearía. Los comentarios se guían solos con el prompt.
    raw = vlm.describe(image_path, model=model or vlm.DEFAULT_MODEL,
                       prompt=prompt, num_predict=45 * n, temperature=0.95,
                       system="")
    # además de las personalidades, se le escapan las reglas del prompt.
    # El PIE no va acá: los comentarios tienen que poder citarlo, es el punto.
    # Que no lo copien enteros lo cuida _sin_copia_del_pie.
    ecos = [p for _, p in elegidos] + [_VOSEO, quien_es_texto(ident),
                                       crudeness_text(crudeness),
                                       TONES.get(tone, TONES[DEFAULT_TONE])]
    got = _sin_copia_del_pie(_sin_eco(_parse(raw), ecos), pie)
    # y por fin, sacar los que dicen casi lo mismo entre sí (y los que repiten
    # lo que ya estaba, si es "más comentarios")
    got = _sin_repetidos(got, previos=[c["text"] for c in ya_estan])
    if not got:
        raise RuntimeError(f"El modelo no devolvió comentarios usables: {raw[:120]}")
    # el i-ésimo comentario es del i-ésimo comentador (se lo pedimos en orden)
    def pulir(t):
        # el modelo a veces le pega una etiqueta de género al final del comentario
        # ("...qué linda! (fem)"). No se la pedimos: se la inventa. Se saca acá.
        t = re.sub(r"\s*[\(\[]\s*(?:fem|hom|masc|f|m|femenino|masculino)\.?\s*[\)\]]\s*$",
                   "", t, flags=re.IGNORECASE).strip()
        t = _rioplatense(t)
        if diminutivos:
            # los piropos (mamita, reinita) SOLO con adulto confirmado; los
            # diminutivos comunes valen para todos
            t = con_diminutivos(t, piropos=not ident["sano"])
        return t
    con_firma = [{"user": u, "text": pulir(t)} for (u, _), t in zip(elegidos, got)]
    todos = ya_estan + con_firma
    _put(name, rel, comments=todos)
    return todos
