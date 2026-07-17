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
TIMEOUT = 180          # en CPU una foto puede tardar bastante
# Abliterado (sin censura) y de 2025: describe mejor y más rápido que llava, que
# además lava las frases de contenido explícito. El desplegable de la web deja
# elegir otro; este es el que se usa si no eligen.
DEFAULT_MODEL = "huihui_ai/qwen3.5-abliterated:4b"

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

Nunca cerrás con moraleja ni con resumen, ni ofrecés ayuda, ni te disculpás."""

# El pie sale en español; BLIP sale en inglés y hay que traducirlo.
PROMPT_IS_SPANISH = True


def caption_prompt(quien=None, voz="", con=(), crudeza="", quien_es=""):
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
# 2. Marca el tono. "Estoy" empuja a describirse ("estoy vestida con..."); "Acá"
#    empuja a situarse ("acá en la playa tomando sol"), que es lo que se busca:
#    la foto ya se ve, el pie la ubica.
PREFILL_PIE = "Acá"


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
