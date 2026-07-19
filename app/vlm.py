"""Descripción de fotos con un modelo de visión (VLM) corriendo en Ollama.

Alternativa a BLIP (app/interrogator.py). La diferencia de fondo: BLIP es chico y
viejo, y le pifia bastante; un VLM moderno mira la foto de verdad y te describe a
la persona (pelo, ropa, pose) en una sola frase, sin preguntas sueltas ni VQA.

A cambio es más lento: en CPU, entre 20 y 40 segundos por foto contra los ~10 de
BLIP. Pensado para el etiquetador de a una, no para pasadas en masa.

Todo local: Ollama corre un servidor en 127.0.0.1:11434. La foto y la frase no
salen de la máquina. El modelo se baja una vez y después anda sin conexión.

OJO con la censura: los modelos de visión oficiales (llava, llama3.2-vision) están
alineados y con contenido explícito te devuelven eufemismos o te rebotan. Para eso
están los finetunes sin censura. Cuál anda mejor lo decide el usuario probando:
por eso el modelo es configurable y no está clavado acá.
"""

import base64
import io
import json
import re
import urllib.error
import urllib.request

from PIL import Image, ImageOps

HOST = "http://127.0.0.1:11434"
TIMEOUT = 300          # en CPU una foto puede tardar bastante (carga en frío + tokens)
# Abliterado (sin censura) y de 2025: describe mejor y más rápido que llava, que
# además lava las frases de contenido explícito. El desplegable de la web deja
# elegir otro; este es el que se usa si no eligen.
DEFAULT_MODEL = "huihui_ai/qwen3.5-abliterated:4b"

# Modelos para ETIQUETAR, en orden de preferencia: primero el q8 (menos
# comprimido, ve mejor, no confunde color de pelo/piel), después el q4 (más
# rápido). Cualquiera de los dos mira mucho mejor que el 4B. Si no hay ninguno,
# cae al DEFAULT. El pie y los comentarios siguen en el abliterado.
TAG_MODELS = [
    # abliterado (SIN censura) + tuneado para NSFW + ve bien (qwen2.5-vl). El
    # mejor para este caso: no rebota lo explícito. En Q4; si querés más precisión
    # de visión hay Q8 del mismo repo.
    "hf.co/mradermacher/Qwen2.5-VL-7B-NSFW-Caption-V3-abliterated-GGUF:Q4_K_M",
    "qwen2.5vl:7b-q8_0", "qwen2.5vl:7b",
]


def tag_model():
    """El mejor modelo de etiquetado que esté bajado (q8 > q4 > default)."""
    try:
        instalados = set(list_models())
        for m in TAG_MODELS:
            if m in instalados:
                return m
        return DEFAULT_MODEL
    except Exception:
        return DEFAULT_MODEL


# Pistas en el nombre de que un modelo es SIN CENSURA (no rebota ni suaviza el
# contenido explícito). Se marcan con 🔞 en el selector para que se vea de una.
_CRUDO_HINTS = ("ablitera", "abliterated", "uncensor", "huihui", "nsfw", "dolphin",
                "lewd", "erotic")


def modelos_descensurados(modelos=None):
    """De la lista de modelos, los que por el nombre son SIN CENSURA."""
    if modelos is None:
        modelos = list_models()
    return [m for m in modelos if any(h in m.lower() for h in _CRUDO_HINTS)]

# Los modelos de visión no leen todos los formatos: un .webp devuelve 400
# ("Failed to load image or audio file"). Mandamos siempre JPEG, así entra
# cualquier cosa que Pillow sepa abrir. De paso achicamos: el modelo igual
# reescala a ~336px, así que mandar 4000px es tiempo tirado.
MAX_SIDE = 1024

# El prompt es lo que decide si describe a la PERSONA o al ambiente. Este pide
# explícitamente pelo/ropa/pose y prohíbe el relleno de fondo, que era la queja
# original con BLIP ("dame más detalle de la persona, no del ambiente").
# El pie se pide DIRECTO EN ESPAÑOL: el tono crudo/vulgar/uruguayo no sobrevive
# al traductor en->es (lo devuelve neutro y planchado). Por eso el motor VLM no
# pasa por translate_es(), a diferencia de BLIP.
#
# Las reglas de ESTILO van acá, en el system, y no en el prompt. Probado: con la
# lista de reglas metida en el prompt, el modelo (un 4b) la toma como plantilla
# de salida y te devuelve un listado con títulos "Ropa:", "Ánimo:", en vez de un
# párrafo. En el system las obedece sin copiarlas.
SYSTEM_PIE = """Escribís el pie de una foto tuya, como el que pondrías en una
red social.

Lo más importante: el que lee YA ESTÁ VIENDO LA FOTO. No se la describas —eso lo
ve solo—. SITUALA: dónde estás y qué estás haciendo. Nada más.

LA ACCIÓN es lo que importa. Si estás haciendo algo (comiendo, tomando sol,
manejando, cocinando), ESO es el pie. Si no estás haciendo nada, como en una
selfie donde solo posás, NO le inventes una acción: decís dónde estás, corto, y
listo. Es mejor un pie de cinco palabras que uno inventado.

Una o dos frases. Corto. Un párrafo, nunca listas ni títulos ni markdown.

Hablás en español uruguayo, de vos. Crudo y directo: las cosas por su nombre,
sin eufemismos ni vueltas.

NO enumerás: ni la ropa, ni el cuerpo, ni el pelo, ni la cara, ni la luz, ni el
fondo. No es un inventario, es un pie de foto. Si algo no se ve, no lo escribas.
Si hay alguien más y te digo quién es, nombralo; si no te lo dije, no lo adivines.

NADA DE POESÍA. Prohibido: metáforas, "el silencio de la tarde", comparar algo
con otra cosa, adornar. Si te digo qué hay confirmado en la foto (etiquetas),
ESO es la verdad de lo que pasa: contalo literal, directo, sin darle vuelta
poética ni inventar un clima que no te dijeron.

Nunca cerrás con moraleja ni con resumen, ni ofrecés ayuda, ni te disculpás."""

# System para SUGERIR ETIQUETAS (no el pie): mira la foto y dice qué hay, palabra
# por palabra, sin vocabulario fijo. Esto es lo que reemplaza a CLIP: CLIP solo
# puede rankear contra etiquetas que YA usaste antes (por eso siempre sugería lo
# mismo); esto mira la foto de verdad y saca lo que ve, aunque sea una palabra
# nueva.
SYSTEM_TAGS = """PRIMERO, SEGURIDAD (por encima de TODO lo demás): si la persona
de la foto parece MENOR de edad —un niño, una niña, un o una adolescente— NO
pongas NINGUNA etiqueta sexual ni del cuerpo. Solo ropa, actividad, edad y lugar,
igual que con un chico. Ante la MÍNIMA duda, tratala como menor. Empezá poniendo
la etiqueta de edad (niño/niña/adolescente) y frená ahí con lo crudo. Esto no se
negocia.

Si es claramente una persona ADULTA, seguí con lo de abajo.

Mirás una foto y anotás, en español RIOPLATENSE, las etiquetas de
lo que VES — grosero y sin pudor, como hablando en la cancha, con la palabra
vulgar que la cosa ES (no la clínica): "tetas" no "senos", "culo" no "glúteos",
"concha" no "vulva". Podés usar el diminutivo si es natural (tetitas, culito),
pero NO inventes palabritas raras para adornar.

REGLA DE ORO: describí solo lo que VES, NO adivines. Pero NO te quedes corto:
recorré la foto entera y sacá CADA cosa real por separado, sin apurarte a parar.
Repasá en orden y anotá lo de cada punto que veas:
1. Sexo y EDAD APARENTE — siempre poné una: adolescente, joven, veinteañera,
   treintañera, madura, cuarentona, milf… la sacás de la cara y el cuerpo, es un
   cálculo aproximado (no la edad exacta, la que aparenta).
2. Pelo (color, largo, si está mojado/atado/suelto, el peinado).
3. Cara y gesto (mirada, boca, expresión).
4. Cuerpo y forma (flaca, gorda, tetona, culona…).
5. CADA prenda por separado y su estado (remera, corpiño, tanga… puesta, corrida,
   abierta). Y lo que falta (en bolas, tetas al aire).
6. Accesorios y marcas: aros, cadenas, anteojos, tatuajes, piercings, lunares.
7. Partes del cuerpo que se ven.
8. La pose (manos, piernas) y el acto.
9. El lugar, solo si es central y evidente.
En una foto con una persona normalmente salen 15 a 25 etiquetas reales — apuntá a
eso, no a 5. Cada una CORTA (1 a 3 palabras), así entran todas.

Lo único prohibido: inventar. No pongas lo que "suele haber" en fotos así (no
todas son latinas, ni de ciudad, ni de rojo): mirá ESTA.
- Colores: solo el que ves CLARO y pegado a una prenda. Si no lo distinguís, no
  pongas color; nunca inventes uno.
- Etnia: solo si es obvia. Si dudás, no la pongas.
- Fondo/lugar: solo si es central y evidente (pileta, playa), nunca de relleno.

Mezclá dos tipos:

SIMPLES — datos sueltos, una o dos palabras:
- Pelo: MIRÁ bien y poné el color y largo que REALMENTE tiene. Si es oscuro es
  morocha, si es claro rubia — no adivines, fijate de verdad en la foto.
- Etnia / piel: solo si es obvia (mirá el tono de piel real, no supongas).
- Sexo y edad aparente (mujer, hombre, pendeja, madura, milf).
- FORMA DEL CUERPO, grosero y sin vueltas: flaca, gorda, gordita, tetona, tetitas,
  culona, culito, culazo, potona, petisa, lomazo, lo que sea.
- Partes que se ven, con el nombre grosero: tetas, culo, concha, pezones, cola,
  pija.
- Ropa prenda por prenda y su estado (corpiño, tanga, tanguita, corrida, abierta,
  transparente) y lo que FALTA con la palabra que es (en bolas, en pelotas, tetas
  al aire, sin bombacha, la concha afuera).

NARRATIVAS — frases cortas de la pose, el gesto o el acto (SOLO si de verdad pasa):
  "mirando a cámara", "mordiéndose el labio", "de espaldas", "en cuatro",
  "piernas abiertas", "mostrando el culo", "abriéndose la concha", "chupando",
  "tocándose"… la actitud y el acto, directo y crudo.

Cada etiqueta tiene que SERVIR: algo que alguien buscaría, o que describe a la
persona, su cuerpo, su ropa o lo que hace. NADA de microdetalles ni pavadas:
- NO "codo levantado", "mano en la cadera", "dedo doblado" y esas boludeces de
  postura mínima. La pose sí, pero la que importa (en cuatro, de espaldas), no un
  gesto suelto.
- NO el color de cualquier cosa suelto ("color violeta", "tono azul"). El color
  va PEGADO a algo que importa (pelo violeta, tanga negra), nunca solo.
- NO objetos de fondo ni decoración que no tengan que ver con la persona.
- NO palabras inventadas ni sílabas sueltas (tata, tita, tuta y esas cosas). Si
  no es una palabra de verdad que describe algo, no va.

OJO CON LOS EJEMPLOS: todas las palabras de arriba (tetas, culo, concha, pija,
aros, tatuajes…) son solo para mostrarte CÓMO se dice (grosero, no clínico). NO
son una lista para copiar. Poné una SOLO si de verdad está en ESTA foto. Si ves
tetas pero no ves la concha, ponés "tetas" y NO "concha". Preferí decir de menos
antes que inventar.

NO opines ("linda", "sexy") y NO inventes lo que no se ve: si no se ve, no va.
Una por línea, minúsculas, 1 a 4 palabras, sin punto final. Si no hay nadie en la
foto, escribís sola la palabra "nadie"."""

# Versión SANA: se usa cuando la persona está etiquetada como menor (misma
# salvaguarda que el pie y los comentarios). Nunca describe cuerpo ni nada
# sexual, pase lo que pase con la perilla.
SYSTEM_TAGS_SANO = """Mirás una foto y anotás, en español, varias etiquetas de lo
que VES: el pelo (color y largo), la ropa (prenda por prenda), la expresión, la
postura, qué está haciendo, y el lugar. Apuntá a unas 12.

Podés mezclar simples ("pelo largo", "campera roja", "sonriendo") con alguna
frase corta de lo que pasa ("jugando en la plaza", "soplando la vela").

Cada etiqueta 1 a 4 palabras, minúsculas, sin punto final, una por línea. NADA
sobre el cuerpo, NADA sexual, nada de opinar ("linda"), nada de inventar lo que
no se ve. Si no hay nadie, escribís sola la palabra "nadie"."""

# El pie sale en español; BLIP sale en inglés y hay que traducirlo.
PROMPT_IS_SPANISH = True


def caption_prompt(quien=None, voz="", con=(), crudeza="", quien_es="", etiquetas=()):
    """El prompt del pie. Corto a propósito: el estilo va en SYSTEM_PIE.

    Sabemos de quién es la foto porque está en su carpeta, así que el pie lo
    escribe ELLA: se lo decimos y no lo adivina. `voz` es cómo habla esa persona
    (se configura en su perfil); vacío = la voz genérica. `con` son los otros
    conocidos que el reconocimiento facial encontró en la foto: así el pie puede
    decir "comiendo con Alberto" en vez de "comiendo con alguien". `crudeza` es
    la perilla de Ajustes (ver posts.CRUDENESS)."""
    quien = quien or "la persona de la foto"
    p = [f"Sos {quien}. Esta es una foto tuya."]
    if quien_es:
        # el pie va en primera persona: sin esto sale "estoy cansado" cuando es
        # una mujer, o al revés
        p.append(quien_es)
    if voz:
        p.append(f"Así hablás vos: {voz}")
    if con:
        quienes = " y ".join(con) if len(con) < 3 else ", ".join(con[:-1]) + f" y {con[-1]}"
        p.append(f"En la foto estás con {quienes}. Nombralo si viene al caso.")
    if etiquetas:
        # Lo etiquetó el usuario (a mano, o la IA mirando la foto): es lo único
        # seguro que hay. El modelo ve mal ("campera" donde hay una blusa) y
        # tiende a florearse; esto es la intención real y le gana a las dos cosas.
        p.append(f"Esto es lo que está pasando en la foto, confirmado: "
                 f"{', '.join(etiquetas)}. ESA es la intención del pie: escribilo "
                 "a partir de eso, literal y directo. No lo enumeres como lista, "
                 "no le busques la vuelta poética, no inventes nada que no esté "
                 "ahí. Si dice \"acostada, sonriendo\", el pie es sobre estar "
                 "acostada y sonriendo — no una metáfora de otra cosa.")
    if crudeza:
        p.append(f"Qué tan subido lo escribís: {crudeza}")
    # Sin esto escribe "Acá tirada con Agus", nombrándose a sí misma como si
    # fuera otra persona (real, con el prefill "Acá" puesto y todo).
    p.append(f"No te nombres a vos: vos SOS {quien}, no estás 'con {quien}'.")
    p.append("Escribí el pie: dónde estás y qué estás haciendo. Corto, una o dos "
             "frases. Si no estás haciendo nada, decilo en pocas palabras y no "
             "inventes.")
    return "\n".join(p)


def _post(path, payload):
    req = urllib.request.Request(
        HOST + path, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return json.loads(r.read().decode("utf-8"))


def is_up():
    """True si Ollama está corriendo. No levanta nada, solo mira."""
    try:
        urllib.request.urlopen(HOST + "/api/tags", timeout=2)
        return True
    except Exception:
        return False


def list_models():
    """Modelos bajados en Ollama. [] si Ollama no está corriendo."""
    try:
        req = urllib.request.Request(HOST + "/api/tags")
        with urllib.request.urlopen(req, timeout=5) as r:
            data = json.loads(r.read().decode("utf-8"))
        return sorted(m["name"] for m in data.get("models", []))
    except Exception:
        return []


def clean_caption(raw):
    """Le saca al pie el envoltorio que el modelo mete igual que le pidas que no.

    Un 4b no se aguanta el formato: encabeza con "**Pie de foto**", mete negritas
    y cursivas, arranca con "Acá va:" y a veces cierra ofreciendo ayuda. El texto
    de adentro suele estar bien, así que se pela en vez de pelear con el prompt.
    """
    t = (raw or "").strip()
    # encabezados tipo "**Pie de foto:**" o "### Pie" en las primeras líneas
    t = re.sub(r"^\s*(#{1,6}\s*)?\**\s*(pie de (la )?foto|pie|caption)\s*:?\**\s*$",
               "", t, flags=re.I | re.M)
    t = re.sub(r"^\s*[-*_]{3,}\s*$", "", t, flags=re.M)      # separadores ---
    t = re.sub(r"\*\*(.+?)\*\*", r"\1", t)                   # negritas
    t = re.sub(r"(?<!\w)[*_](.+?)[*_](?!\w)", r"\1", t)      # cursivas
    t = re.sub(r"^\s*>\s?", "", t, flags=re.M)               # citas
    # Preámbulos, solo los conocidos. Nada de "cortar hasta los dos puntos": el
    # pie empieza con "Estoy" (ver PREFILL_PIE) y una frase como "Estoy parada:
    # mirando" perdería el arranque.
    t = re.sub(r"^\s*(acá|aca|aquí|aqui) (va|tenés|tenes|te dejo)[^:\n]{0,40}:\s*",
               "", t, flags=re.I)
    t = re.sub(r"^\s*(yo,?\s+\w+,?\s+en mi propia voz|en mi propia voz)\s*:\s*",
               "", t, flags=re.I)
    return "\n".join(l.strip() for l in t.splitlines() if l.strip()).strip()


def _encode(image_path):
    """La foto como JPEG en base64: normaliza formato, rotación y tamaño."""
    im = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
    if max(im.size) > MAX_SIDE:
        im.thumbnail((MAX_SIDE, MAX_SIDE))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# Primera palabra del pie. Hace dos cosas de una:
# 1. Fuerza la primera persona. Pedírselo NO alcanza: probado, se va a "Se ve a
#    Agus..." o "Veo un cuerpo de mujer..." una de cada tres veces. Si la primera
#    palabra se la escribimos nosotros, no tiene por dónde irse.
# 2. Marca el tono situacional: "acá/hoy/recién comiendo..." en vez de describir.
#
# Es una LISTA y no una sola palabra porque si le clavás siempre "Acá", TODOS los
# pies arrancan igual ("Acá comiendo", "Acá abajo del árbol"...). Rotando, cada
# foto abre distinto. Todas mantienen la primera persona situacional.
PREFILL_PIE = "Acá"          # el default suelto (cuando no se rota)
PREFILL_OPCIONES = ["Acá", "Hoy", "Recién", "Ahora", "Justo acá", "En la foto",
                    "Este día", "Un rato"]


def describe(image_path, model=DEFAULT_MODEL, prompt=None, num_predict=280,
             temperature=0.8, system=SYSTEM_PIE, prefill=""):
    """Lo que el modelo tenga para decir de la foto, según el prompt.

    Sin prompt: el pie de foto, en español. Con otro prompt sirve para cualquier
    cosa — de eso se aprovecha app/comments.py.

    El estilo va en `system`, no en el prompt: probado, un 4b toma la lista de
    reglas del prompt como plantilla de salida y devuelve un listado con títulos
    ("Ropa:", "Ánimo:") en vez de un párrafo. En el system las obedece.

    num_predict alto por defecto porque el pie es un párrafo mediano o largo:
    con 80 tokens se corta a la mitad de la frase. Ojo que a ~3,5 tok/s cada
    token se paga en segundos.

    `prefill` es la primera palabra de la respuesta: se la escribimos nosotros y
    el modelo continúa desde ahí, así no puede irse de la voz. Va por /api/chat,
    que es el único que acepta empezarle la respuesta.

    Lanza RuntimeError con un mensaje legible si algo falla."""
    prompt = prompt or caption_prompt()
    try:
        b64 = _encode(image_path)
    except Exception as e:
        raise RuntimeError(f"No se pudo leer la imagen: {e}")
    # Los modelos de razonamiento (Qwen y compañía) gastan todos los tokens
    # "pensando" y devuelven la respuesta vacía: think=False. Los que no razonan
    # (llava) ignoran esa clave sin quejarse.
    # temperatura: baja para describir (literal), alta para inventar.
    opts = {"temperature": temperature, "num_predict": num_predict}
    if prefill:
        msgs = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt, "images": [b64]},
            {"role": "assistant", "content": prefill}]
        payload = ("/api/chat", {"model": model, "messages": msgs,
                                 "stream": False, "think": False, "options": opts})
    else:
        payload = ("/api/generate", {"model": model, "prompt": prompt,
                                     "images": [b64], "system": system or "",
                                     "stream": False, "think": False, "options": opts})
    try:
        r = _post(*payload)
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:200]
        if e.code == 404:
            raise RuntimeError(
                f"Ollama no tiene el modelo '{model}'. Bajalo con: ollama pull {model}")
        raise RuntimeError(f"Ollama contestó {e.code}: {detail}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"No se pudo hablar con Ollama en {HOST}: {e.reason}. "
                           "¿Está corriendo?")
    if prefill:
        # /api/chat devuelve la continuación; la primera palabra la pusimos
        # nosotros y hay que volver a pegarla adelante
        sigue = (r.get("message", {}).get("content") or "").strip()
        return f"{prefill} {sigue}".strip() if sigue else ""
    return (r.get("response") or "").strip()


def tags_prompt(quien=None, evitar=()):
    """El prompt del sugeridor de etiquetas. El estilo (formato, cantidad, nada
    de inventar) va en SYSTEM_TAGS; acá solo el pedido puntual.

    `evitar` son las etiquetas que ya salieron (botón "sugerir más"): se le pide
    al modelo que NO las repita y busque otras cosas distintas en la foto."""
    base = (f"Esta es una foto de {quien}. Poné las etiquetas de lo que ves."
            if quien else "Poné las etiquetas de lo que ves en esta foto.")
    if evitar:
        # "Sugerir más" / rondas del auto-continuo: NO le damos una lista de
        # categorías (eso lo PRIMA a inventar aros/pies/etc. que no están). Solo
        # le pedimos otras que REALMENTE se vean, con salida explícita si no queda
        # nada — así para en vez de alucinar.
        base += (" Ya salieron estas, NO las repitas: " + ", ".join(evitar) + ". "
                 "Mirá de nuevo y poné SOLO otras cosas que estén DE VERDAD en la "
                 "foto y no hayas dicho. Si ya nombraste todo lo que se ve, NO "
                 "inventes nada: escribí una sola palabra, \"nada\", y listo.")
    return base


# Diccionario "endurecedor": si el modelo se pone pudoroso y devuelve la palabra
# clínica/suave, se la cambia por la grosera rioplatense. Solo cambia vocabulario
# (senos -> tetas); NO inventa ni agrega nada que el modelo no haya visto. Se
# aplica palabra por palabra, así "senos grandes" -> "tetas grandes". Se eligen
# solo términos clínicos inequívocos (nada de "cola"/"pecho", que son ambiguos).
_CRUDO_MAP = {
    "senos": "tetas", "pechos": "tetas", "busto": "tetas", "mamas": "tetas",
    "seno": "teta", "pezon": "pezón",
    "gluteos": "culo", "glúteos": "culo", "nalgas": "culo", "trasero": "culo",
    "vulva": "concha", "vagina": "concha", "genitales": "concha",
    "pene": "pija", "falo": "pija",
    "desnuda": "en bolas", "desnudo": "en bolas", "desnudez": "en bolas",
}


def _conversiones():
    """Conversiones EDITABLES por el usuario (desde el probador): {tal_cual_lo_dice
    el modelo: como lo queremos}. A diferencia de _CRUDO_MAP, matchea la etiqueta
    ENTERA (no palabra por palabra), para desambiguar: p.ej. "rusa" -> "paja rusa"
    (si no, el pie habla del país). Se guardan en config (f_conversiones)."""
    try:
        from app.config import load_config
        c = load_config().get("f_conversiones", {}) or {}
        return {str(k).strip().lower(): str(v).strip()
                for k, v in c.items() if str(k).strip() and str(v).strip()}
    except Exception:
        return {}


def _aplicar_conversiones(tags, conv=None):
    """Cambia cada etiqueta que esté EXACTA en el mapa de conversiones."""
    if conv is None:
        conv = _conversiones()
    if not conv:
        return tags
    out, seen = [], set()
    for t in tags:
        t2 = conv.get(t.lower(), t)
        if t2.lower() not in seen:
            seen.add(t2.lower())
            out.append(t2)
    return out


# Vocabulario CRUDO de confianza: la jerga/diminutivos/descriptores que un
# diccionario formal NO tiene pero nosotros SÍ queremos. Junto con las etiquetas
# que el usuario ya usó (all_tags), forma el "diccionario" contra el que se marca
# una palabra como conocida o nueva/dudosa (no se borra ninguna: ver la web).
# Estirable a mano.
VOCAB_CRUDO = {
    # cuerpo / grosero / diminutivos
    "tetas", "teta", "tetona", "tetitas", "tetota", "lolas", "lola",
    "culo", "culito", "culona", "culazo", "culón", "orto", "potona", "potota",
    "cola", "colita", "concha", "conchita", "chocho", "pija", "pito",
    "pezones", "pezón", "pezoncitos", "flaca", "flaco", "gorda", "gordo",
    "gordita", "rellenita", "petisa", "petiso", "alta", "bajita", "lomazo",
    "fornido", "musculoso", "tonificada",
    # pelo / etnia / sexo / edad
    "rubia", "rubio", "morocha", "morocho", "castaña", "pelirroja", "canosa",
    "lacio", "rulos", "rizado", "ondulado", "flequillo", "colorada",
    "latina", "morena", "moreno", "trigueña", "negra", "negro", "asiática",
    "asiático", "blanca", "blanco", "mulata", "india",
    "mujer", "hombre", "pendeja", "pendejo", "madura", "maduro", "milf",
    "joven", "adolescente", "adulta", "adulto", "vieja", "viejo",
    # ropa / estado
    "corpiño", "tanga", "tanguita", "bombacha", "bikini", "malla", "lencería",
    "vestido", "pollera", "remera", "musculosa", "campera", "jean", "short",
    "medias", "botas", "tacos", "escote", "transparente", "mojada", "húmeda",
    # actitud / pose sueltas de una palabra
    "sonriendo", "posando", "acostada", "parada", "sentada", "agachada",
    "arrodillada", "desnuda", "semidesnuda", "provocativa",
    # actos (jerga; algunos los tira el clasificador de posiciones)
    "chupando", "mamando", "cogiendo", "penetración", "rusa", "paja",
    "dedeando", "sola", "masturbándose", "en cuatro",
}


# Palabras genéricas que el modelo mete adelante y hacen que "latina" y "mujer
# latina" parezcan distintas. Para deduplicar, se sacan del frente: las dos caen
# a la misma clave "latina".
_GENERICOS = {"mujer", "hombre", "chica", "chico", "piba", "pibe", "mina",
              "persona", "una", "un", "la", "el", "de"}


def _dup_key(t):
    """Clave para deduplicar cuasi-repetidas: saca las palabras genéricas del
    frente. "mujer latina" -> "latina"; "una rubia" -> "rubia". Deja al menos
    una palabra (así "mujer" solo sigue siendo "mujer")."""
    ws = t.split()
    while len(ws) > 1 and ws[0] in _GENERICOS:
        ws = ws[1:]
    return " ".join(ws)


def _dedup_key(tags):
    """Dedup por _dup_key, conservando el primero de cada grupo y el orden."""
    seen, out = set(), []
    for t in tags:
        k = _dup_key(t)
        if k not in seen:
            seen.add(k)
            out.append(t)
    return out


def _endurecer(tags):
    """Pasa cada etiqueta por _CRUDO_MAP (clínico -> grosero). Si un reemplazo
    genera repetidas (había "senos" y "tetas"), se deduplica."""
    out = []
    for t in tags:
        for suave, crudo in _CRUDO_MAP.items():
            t = re.sub(rf"\b{re.escape(suave)}\b", crudo, t)
        out.append(t)
    seen, dedup = set(), []
    for t in out:
        if t not in seen:
            seen.add(t)
            dedup.append(t)
    return dedup


# Basura que el modelo tira igual aunque el prompt se lo prohíba: el color suelto
# ("color violeta", "tono azul") y las etiquetas con prefijo "detalle:". El color
# que importa va pegado a algo (pelo violeta) y ese NO empieza con "color".
# Prefijos de "campo: valor" o de categorías que el modelo tira cuando se pone a
# rellenar/adivinar: casi siempre son alucinación o formato roto. "latina" plana
# sí pasa (puede ser real); "etnia: latina", "fondo urbano", "color rojo" no.
_RUIDO_RE = re.compile(
    r"^(color(es)?|tono|detalle|fondo|etnia|ethnia|lugar|ambiente|escenario|entorno)\b")

# Balbuceo confirmado del 4b: sílabas sueltas que no son palabras. NO se puede
# filtrar por forma (tata y "teta"/"nena" son iguales estructuralmente y esas SÍ
# valen), así que es una lista de coincidencia exacta. Estirable a mano.
_BASURA = {"tata", "tato", "tita", "tito", "tute", "tuta", "tutu", "toto",
           "tete", "titi"}


def _clean_one(line):
    """Una línea cruda del modelo -> una etiqueta limpia, o None si no sirve.
    Saca viñetas/numeración/comillas, pasa a minúsculas, descarta vacías, las
    de 'nadie', las de más de 4 palabras (ya es una frase) y la basura conocida
    (color suelto, 'detalle:')."""
    t = line.strip()
    t = re.sub(r"^[-*•]\s*", "", t)
    t = re.sub(r"^\d+[.)]\s*", "", t)
    t = t.strip(" \"'.,;:").lower()
    if not t or t in ("nadie", "no hay nadie", "ninguna", "nada", "n/a", "ninguno"):
        return None
    if t in _BASURA:              # balbuceo suelto (tata, tita, tuta…)
        return None
    if len(t.split()) > 5:        # más de 5 palabras ya es una frase, no etiqueta
        return None
    if _RUIDO_RE.match(t):        # "color violeta", "tono azul", "detalle: ..."
        return None
    return t


def _parse_tags(raw):
    """Del texto crudo del modelo a una lista de etiquetas cortas y limpias.
    Tolera que el modelo devuelva una por línea O varias separadas por coma."""
    seen, dedup = set(), []
    for line in (raw or "").splitlines():
        for pedazo in re.split(r"[,;]", line):
            t = _clean_one(pedazo)
            if t and t not in seen:
                seen.add(t)
                dedup.append(t)
    return dedup[:60]


def suggest_tags(image_path, model=DEFAULT_MODEL, quien=None, crudo=True, evitar=()):
    """Etiquetas de lo que el modelo VE en la foto, en español, sin vocabulario
    fijo (a diferencia de CLIP, que solo puede rankear contra etiquetas que ya
    usaste antes). Cada corrida puede sacar palabras nuevas si la foto las tiene.

    `crudo=True` (por defecto): mira TODO, incluida ropa/pose/acto explícito.
    `crudo=False`: versión sana, sin cuerpo ni nada sexual — la usa el llamador
    cuando la persona está etiquetada como menor (misma salvaguarda que el pie).
    `evitar`: etiquetas ya salidas, para el botón "sugerir más" — pide otras y
    filtra por las dudas si el modelo igual repite. Sube la temperatura para que
    varíe más en las siguientes tandas.

    Lanza RuntimeError con mensaje legible si algo falla (ver describe())."""
    system = SYSTEM_TAGS if crudo else SYSTEM_TAGS_SANO
    # temp baja en la 1ra pasada (fiel), alta en "sugerir más" (escarba, no repite)
    raw = describe(image_path, model=model, prompt=tags_prompt(quien, evitar),
                   system=system, num_predict=600 if crudo else 220,
                   temperature=0.4 if evitar else 0.25)
    tags = _parse_tags(raw)
    if crudo:
        # si el modelo se puso pudoroso y tiró la clínica, la endurecemos
        tags = _endurecer(tags)
    tags = _dedup_key(tags)                      # "latina" vs "mujer latina"
    tags = _aplicar_conversiones(tags)           # rusa -> paja rusa (editable)
    if evitar:
        ev = {_dup_key(e.strip().lower()) for e in evitar}
        tags = [t for t in tags if _dup_key(t) not in ev]
    return tags


def suggest_tags_debug(image_path, model=DEFAULT_MODEL, quien=None, crudo=True,
                       evitar=()):
    """Para el PROBADOR: devuelve las tres capas para que el usuario vea qué
    entiende el modelo y cómo lo transformamos:
      - crudo: el texto tal cual lo tiró el modelo.
      - limpias: después de parsear/endurecer/dedup (lo que iría a la lista).
      - final: después de aplicar las conversiones editables (rusa->paja rusa).

    `evitar`: etiquetas ya guardadas (para "Guardar y seguir") — pide OTRAS y
    filtra por las dudas si el modelo igual repite. Sube la temperatura para que
    escarbe y varíe."""
    system = SYSTEM_TAGS if crudo else SYSTEM_TAGS_SANO
    # PROBADOR: tope de tokens más bajo (300) que el visor. En CPU responde bastante
    # más rápido; si querés más, das "Etiquetar" de nuevo (traés otra tanda).
    raw = describe(image_path, model=model, prompt=tags_prompt(quien, evitar),
                   system=system, num_predict=300 if crudo else 220,
                   temperature=0.4 if evitar else 0.25)
    limpias = _parse_tags(raw)
    if crudo:
        limpias = _endurecer(limpias)
    limpias = _dedup_key(limpias)
    final = _aplicar_conversiones(limpias)
    if evitar:
        ev = {_dup_key(e.strip().lower()) for e in evitar if str(e).strip()}
        final = [t for t in final if _dup_key(t) not in ev]
    return {"crudo": raw, "limpias": limpias, "final": final}


def suggest_tags_pairs(image_path, model=DEFAULT_MODEL, quien=None, crudo=True,
                       evitar=()):
    """Para el PROBADOR (vista de dos columnas): por cada etiqueta devuelve
    {orig, final} donde:
      - orig  = la palabra TAL CUAL la tiró el modelo (limpia de viñetas), para
                que el usuario VEA qué vio de verdad y no corrija a ciegas.
      - final = nuestra interpretación (endurecido clínico->grosero + conversión
                editable rusa->paja rusa); es lo editable/guardable.
    Así, si en `final` aparece algo raro, `orig` muestra de dónde salió.

    `evitar`: etiquetas (finales) ya guardadas — trae OTRAS de la misma foto."""
    system = SYSTEM_TAGS if crudo else SYSTEM_TAGS_SANO
    # tope alto: la primera pasada tiene que traer TODO (el modo lote no re-analiza
    # la misma foto, así que exprimimos una sola tirada bien larga).
    raw = describe(image_path, model=model, prompt=tags_prompt(quien, evitar),
                   system=system, num_predict=450 if crudo else 260,
                   temperature=0.4 if evitar else 0.25)
    crudas = _parse_tags(raw)                     # lo que dijo el modelo (col. izq.)
    # endurecer clínico->grosero PERO manteniendo el par (orig, endurecida)
    trabajo = []
    for c in crudas:
        h = c
        if crudo:
            for suave, cr in _CRUDO_MAP.items():
                h = re.sub(rf"\b{re.escape(suave)}\b", cr, h)
        trabajo.append((c, h))
    # dedup cuasi-repetidas por la clave de la endurecida, conservando el primero
    seen, ded = set(), []
    for c, h in trabajo:
        k = _dup_key(h)
        if k not in seen:
            seen.add(k)
            ded.append((c, h))
    # conversión editable (match de etiqueta entera) sobre la endurecida
    conv = _conversiones()
    ev = {_dup_key(e.strip().lower()) for e in evitar if str(e).strip()}
    seenf, pares = set(), []
    for c, h in ded:
        f = conv.get(h.lower(), h)
        kf = _dup_key(f.lower())
        if kf in seenf or kf in ev:
            continue
        seenf.add(kf)
        pares.append({"orig": c, "final": f})
    if not pares:
        # diagnóstico: ¿el modelo NO dijo nada, o lo dijo y lo filtramos todo?
        motivo = "el modelo no devolvió texto" if not (raw or "").strip() else \
                 f"dijo algo pero se filtró/dedup todo (crudas={len(crudas)})"
        print(f"[PAIRS] 0 etiquetas · {motivo} · raw[:120]={ (raw or '')[:120]!r }")
    return pares


def suggest_tags_stream(image_path, model=DEFAULT_MODEL, quien=None, crudo=True,
                        evitar=()):
    """Igual que suggest_tags pero GENERADOR: va largando cada etiqueta apenas el
    modelo la termina, en vez de esperar toda la respuesta. La mirada del modelo
    (lo caro) se paga una sola vez; las etiquetas salen en vivo mientras escribe.

    Rinde etiquetas ya limpias, endurecidas (si crudo) y sin repetir. Lanza
    RuntimeError legible si Ollama falla."""
    system = SYSTEM_TAGS if crudo else SYSTEM_TAGS_SANO
    b64 = _encode(image_path)
    # 1ra pasada: temp baja = fiel a la foto. "Sugerir más": temp ALTA para que
    # escarbe y no repita lo mismo (si no, devuelve lo obvio y el filtro lo vacía).
    # num_predict amplio: qwen2.5vl no balbucea, y queremos que llegue a 15-25.
    opts = {"temperature": 0.4 if evitar else 0.25,
            "num_predict": 600 if crudo else 220}
    payload = {"model": model, "prompt": tags_prompt(quien, evitar),
               "images": [b64], "system": system, "stream": True,
               "think": False, "options": opts}
    req = urllib.request.Request(
        HOST + "/api/generate", data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    ev = {_dup_key(e.strip().lower()) for e in evitar}
    seen = set()
    stats = {"crudas": 0, "pasaron": 0}     # para el log: cuántas tiró vs pasaron
    conv = _conversiones()                  # rusa -> paja rusa (editable, una vez)

    def emitir(linea):
        """Una línea cruda -> 0+ etiquetas nuevas listas para mostrar. Tolera
        coma (el modelo a veces separa con coma en vez de por línea) y deduplica
        por _dup_key, así "latina" y "mujer latina" no salen las dos."""
        for pedazo in re.split(r"[,;]", linea):
            if pedazo.strip():
                stats["crudas"] += 1
            t = _clean_one(pedazo)
            if not t:
                continue
            if crudo:
                for suave, cr in _CRUDO_MAP.items():
                    t = re.sub(rf"\b{re.escape(suave)}\b", cr, t)
            t = conv.get(t.lower(), t)      # conversión editable (rusa -> paja rusa)
            k = _dup_key(t)
            if k in seen or k in ev:
                continue
            seen.add(k)
            stats["pasaron"] += 1
            yield t

    buf = ""
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            for raw in r:                       # Ollama manda NDJSON, una por línea
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    obj = json.loads(raw)
                except ValueError:
                    continue
                buf += obj.get("response", "")
                # emitir cada renglón COMPLETO (deja en buf lo que sigue después
                # de la última coma/salto, por si la etiqueta viene cortada)
                while "\n" in buf:
                    linea, buf = buf.split("\n", 1)
                    for t in emitir(linea):
                        yield t
                if obj.get("done"):
                    break
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:200]
        if e.code == 404:
            raise RuntimeError(
                f"Ollama no tiene el modelo '{model}'. Bajalo con: ollama pull {model}")
        raise RuntimeError(f"Ollama contestó {e.code}: {detail}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"No se pudo hablar con Ollama en {HOST}: {e.reason}. "
                           "¿Está corriendo?")
    for t in emitir(buf):                       # el último pedazo sin \n final
        yield t
    # log en la terminal: cuántas "cosas" tiró el modelo vs cuántas te llegaron
    # (si la brecha es grande, el filtro se está comiendo etiquetas)
    print(f"[TAGS] modelo tiró ~{stats['crudas']} · pasaron {stats['pasaron']}"
          f" (evitar={len(ev)}, temp={'alta' if evitar else 'baja'})", flush=True)
