"""Pies de foto armados con MOLDES tuyos, sin modelo.

Por qué existe: probamos seis configuraciones de LLM local para el pie (4B y 7B,
con y sin arranque, con y sin imagen, reglamento largo y prompt liviano). Ninguna
escribió una frase que el usuario hubiera firmado: o describía la foto como un
inventario, o recitaba el propio prompt, o sonaba a folleto en español neutro. Y
tardaba entre 30 y 230 segundos. Un modelo de ese tamaño en CPU sabe MIRAR (el
etiquetado anda bien) pero no sabe sonar como vos.

Entonces se reparte el trabajo: el modelo mira y etiqueta, y la escritura la
ponés vos UNA vez en moldes que se combinan. Sale instantáneo, siempre con
sentido y con tu voz. Mismo trato que app/phrases.py hizo con las frases de BLIP.

CÓMO ARMA EL PIE
Un pie es un CUERPO (de qué va la foto) y a veces un CIERRE (el ánimo, corto).
Los cuerpos se disparan por etiqueta; los cierres son genéricos y pegan con
cualquiera, así 60 cuerpos × 20 cierres dan más de mil pies con 80 frases
escritas. Sin el cierre, para que no salgan todos con la misma cadencia.

SOBRE QUÉ SE DISPARAN
Sobre SITUACIÓN y LUGAR (playa, mate, recién levantada), nunca sobre atributos
(pelo castaño, flaca, alta). Nadie postea "tengo el pelo castaño": eso sirve para
buscar, no para el pie. Los atributos quedan afuera a propósito.

data/moldes.json = {"cuerpos": [{"tags": [...], "texto": "...", "nsfw": bool}],
                    "cierres": [{"texto": "...", "nsfw": bool}],
                    "alias": {"denuda": "desnuda", ...}}
Vacío o sin archivo = los de fábrica de acá abajo.
"""

import hashlib
import json
import re
import unicodedata

from app import DATA_DIR
from app import vlm

MOLDES_PATH = DATA_DIR / "moldes.json"

# Cada cuerpo se dispara si la foto tiene ALGUNA de sus tags. Los de nsfw=True
# solo entran con adulto confirmado (ver posts.identidad -> sano).
CUERPOS_FABRICA = [
    # --- playa, pileta, agua
    (["playa"], "Un día más de playa y no me quiero ir más.", False),
    (["playa"], "Me estoy quemando viva pero no me pienso mover.", False),
    (["pileta"], "Todo el día en la pileta, no hay plan mejor.", False),
    (["pileta"], "Me tiré al agua y estaba helada, casi me muero.", False),
    (["arena"], "Tengo arena hasta en lugares que ni sabía.", False),
    # --- sol y calor
    (["sol"], "Tirada al sol sin hacer nada, como corresponde.", False),
    (["calor"], "Me derrito. No doy más con este calor.", False),
    # --- mate, comida, cocina
    (["mate"], "Mate y nada más. Así se arregla cualquier día.", False),
    (["mate"], "Ya voy por el cuarto mate y no pienso parar.", False),
    (["cocinando"], "Cocinando algo que ojalá salga bien.", False),
    (["comiendo"], "Comiendo como si no hubiera un mañana.", False),
    (["asado"], "Asado en curso. No me hablen hasta que termine.", False),
    # --- casa, cama, despertar
    (["cama"], "No me pienso levantar en todo el día.", False),
    (["recién levantada"], "Recién me levanto y ya quiero volver.", False),
    (["siesta"], "La siesta es sagrada, no se discute.", False),
    (["sillón"], "Sillón, manta y nada de planes.", False),
    (["pijama"], "En pijama desde ayer y no me arrepiento.", False),
    (["en casa"], "Domingo de no hacer absolutamente nada.", False),
    # --- baño, arreglarse
    (["baño"], "Media hora acá adentro y todavía no salgo.", False),
    (["ducha"], "Recién salgo del agua y ya extraño la ducha.", False),
    (["espejo"], "Foto al espejo porque no había nadie para sacármela.", False),
    (["maquillaje"], "Una hora arreglándome para salir dos.", False),
    (["pelo mojado"], "Con el pelo mojado y sin ganas de secarlo.", False),
    # --- salir
    (["salida"], "Arrancando la noche. A ver cómo termina esto.", False),
    (["fiesta"], "Se armó y no la vimos venir.", False),
    (["vestido"], "Me puse esto y ya no me lo saco más.", False),
    (["de noche"], "Una noche de esas que no se planean.", False),
    (["cerveza"], "La primera del día, prometo que es la única.", False),
    # --- afuera
    (["caminando"], "Caminando sin rumbo, que es mi plan favorito.", False),
    (["auto"], "Al volante y con la música al mango.", False),
    (["viaje"], "Lejos de todo y sin ganas de volver.", False),
    (["campo"], "Verde por todos lados y cero señal. Perfecto.", False),
    (["lluvia"], "Llueve y yo acá, mirando por la ventana.", False),
    (["frío"], "Muerta de frío pero valió la pena salir.", False),
    (["atardecer"], "Me quedé mirando hasta que se hizo de noche.", False),
    # --- actividad
    (["gimnasio"], "Sobreviví al entrenamiento. Apenas.", False),
    (["transpirada"], "Hecha pomada pero contenta.", False),
    (["bailando"], "Me solté y ya no hubo forma de pararme.", False),
    (["trabajando"], "Laburando cuando debería estar durmiendo.", False),
    (["leyendo"], "Metida en este libro desde hace horas.", False),
    # --- compañía y ánimo (sin lugar)
    (["amigas"], "Con estas se me pasa el día volando.", False),
    (["sonriendo"], "Me agarraron riéndome y quedó esta.", False),
    (["mirando a cámara"], "Me sacaron sin avisar y salió esta.", False),
    (["expresión neutral"], "No estaba de humor para posar.", False),
    (["mascota"], "El verdadero protagonista de la foto no soy yo.", False),
    # combinaciones: piden las DOS y por eso le ganan a las de arriba
    (["playa", "sol"], "Playa, sol y cero obligaciones. No pido más.", False),
    (["cama", "recién levantada"], "Me desperté hace rato y sigo acá.", False),
    (["espejo", "sonriendo"], "Me reí sola sacándome esta.", False),

    # --- subidas de tono: solo con adulto confirmado
    (["desnuda"], "Sin nada puesto, que es como mejor estoy.", True),
    (["lencería"], "Me compré esto y no aguanté a estrenarlo.", True),
    (["bikini"], "Este bikini me tapa menos de lo que debería.", True),
    (["tetas"], "Al final las saqué, hacía mucho calor.", True),
    (["culo"], "Me dijeron que de atrás salgo mejor.", True),
    (["mojada"], "Salí del agua y quedó todo pegado al cuerpo.", True),
    (["sexy"], "Sabía perfectamente lo que estaba haciendo.", True),
    (["masturbación"], "Estaba sola y me di un gusto.", True),
    (["piernas abiertas"], "Me puse cómoda y quedó esto.", True),
    (["de cuatro"], "Me acomodé así y alguien aprovechó para sacarme una.", True),
    # combinaciones subidas: piden las dos, así no afirman lo que no está etiquetado
    (["desnuda", "sola"], "Sola en casa y sin ganas de vestirme.", True),
    (["desnuda", "cama"], "En la cama, sin nada, y sin apuro por levantarme.", True),
    (["desnuda", "ducha"], "Se me hizo larga la ducha hoy.", True),
    (["desnuda", "playa"], "Nadie a la vista, así que sobraba la malla.", True),
]

# Cierres: van pegados a cualquier cuerpo, sin depender de etiquetas. Cortos, que
# el pie no se vuelva un párrafo.
CIERRES_FABRICA = [
    ("Un lujo.", False),
    ("No pido más nada.", False),
    ("Así da gusto.", False),
    ("Después me hago cargo.", False),
    ("Mañana veo.", False),
    ("Ni ganas de moverme.", False),
    ("Necesitaba esto.", False),
    ("Se me fue el día.", False),
    ("Otra vez.", False),
    ("Como debe ser.", False),
    ("No me juzguen.", False),
    ("Y todavía es temprano.", False),
    ("Qué se yo.", False),
    ("Igual no me arrepiento.", False),
    ("Todo bien por acá.", False),
    ("Mirá si me voy a quejar.", True),
    ("Alguien va a tener que hacerse cargo.", True),
    ("A ver quién se anima.", True),
    ("Ya sé lo que están pensando.", True),
    ("No me hagan hablar.", True),
]

# Tus etiquetas vienen con tipeos y sinónimos ("denuda", "en bolas"). Sin esto un
# molde escrito para "desnuda" nunca se prendería en esas fotos. Salió de mirar
# etiquetas reales: 'denuda', 'ojoz azules', 'len cuatro' estaban en la base.
ALIAS_FABRICA = {
    # tipeos y sinónimos de lo subido
    "denuda": "desnuda", "desnudo": "desnuda", "en bolas": "desnuda",
    "en pelotas": "desnuda", "encuerada": "desnuda", "sin ropa": "desnuda",
    "torso en bolas": "tetas", "en tetas": "tetas", "tetitas": "tetas",
    "tetona": "tetas", "tetas al aire": "tetas", "tetas redondas": "tetas",
    "tetas perfectas": "tetas", "pezones": "tetas",
    "culito": "culo", "culina": "culo", "culazo": "culo",
    "mostrando el culito": "culo", "de espaldas": "culo",
    "len cuatro": "de cuatro", "de 4": "de cuatro",
    "lomazo": "sexy", "provocativa": "sexy", "cuerpo perfecto": "sexy",
    "paja": "masturbación", "abierta": "piernas abiertas",
    "ropa interior": "lencería", "bombacha": "lencería", "corpiño": "lencería",
    # sinónimos de lugar y situación
    "mar": "playa", "piscina": "pileta", "tomando sol": "sol",
    "acostada": "cama", "en la cama": "cama", "durmiendo": "siesta",
    "sofa": "sillón", "sofá": "sillón", "entrenando": "gimnasio",
    "boliche": "salida", "saliendo": "salida", "cumpleaños": "fiesta",
    "birra": "cerveza", "manejando": "auto", "viajando": "viaje",
    "pasto": "campo", "naturaleza": "campo", "lloviendo": "lluvia",
    "puesta de sol": "atardecer", "libro": "leyendo", "trabajo": "trabajando",
    "cocina": "cocinando", "comida": "comiendo", "bañándose": "ducha",
    "con amigas": "amigas", "perro": "mascota", "gato": "mascota",
    "selfie": "espejo", "maquillándose": "maquillaje", "casa": "en casa",
    "seria": "expresión neutral", "despeinada": "recién levantada",
    "noche": "de noche", "calle": "caminando",
}


def _sin_tildes(s):
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn")


def _norm(t):
    """Compara etiquetas sin tildes ni puntuación: "recién" y "recien" son la
    misma, y así los tipeos de acentos no rompen el disparo del molde."""
    return " ".join(_sin_tildes(re.sub(r"[^\w\s]", " ", (t or "").lower())).split())


def _load():
    if not MOLDES_PATH.is_file():
        return {}
    try:
        with open(MOLDES_PATH, encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def _save(d):
    MOLDES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MOLDES_PATH, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)


def cuerpos():
    """La lista PLANA que usa el motor: [(tags, texto, nsfw)], un texto por tupla.

    Es la forma interna. En el JSON un mismo molde puede venir como `texto` (una
    sola frase, formato viejo) o como `textos` (un array de variantes para la
    misma combinación, formato nuevo): las dos se aplanan acá a una tupla por
    frase, así el resto del motor no se entera del cambio."""
    d = _load().get("cuerpos")
    if not (isinstance(d, list) and d):
        return CUERPOS_FABRICA
    plano = []
    for c in d:
        if not isinstance(c, dict):
            continue
        tags = c.get("tags", [])
        nsfw = bool(c.get("nsfw"))
        # `textos` (array) gana; si no, el `texto` suelto de siempre
        variantes_ = c.get("textos") if isinstance(c.get("textos"), list) else None
        if variantes_ is None:
            variantes_ = [c.get("texto", "")]
        for x in variantes_:
            x = (x or "").strip()
            if tags and x:
                plano.append((tags, x, nsfw))
    return plano or CUERPOS_FABRICA


def combinaciones(planos=None):
    """La lista PLANA agrupada por combinación: [(tags, [textos], nsfw, borrador)].
    Es lo que ve el editor y lo que se exporta — una fila por combinación, con
    todas sus variantes juntas. Junta las tuplas que comparten las mismas
    etiquetas (normalizadas) y el mismo nsfw, conservando el orden de aparición.

    `borrador` (lo generó el modelo, falta revisar) se saca de las entradas crudas
    del JSON, no de `planos` — por eso se lee acá aparte."""
    if planos is None:
        planos = cuerpos()
    # qué combinaciones están marcadas borrador en el JSON
    marcados = set()
    for c in (_load().get("cuerpos") or []):
        if isinstance(c, dict) and c.get("borrador"):
            marcados.add((tuple(sorted(_norm(t) for t in c.get("tags", []))), bool(c.get("nsfw"))))
    orden, grupos = [], {}
    for tags, texto, nsfw in planos:
        clave = (tuple(sorted(_norm(t) for t in tags)), nsfw)
        if clave not in grupos:
            grupos[clave] = [list(tags), [], nsfw, clave in marcados]
            orden.append(clave)
        if texto not in grupos[clave][1]:
            grupos[clave][1].append(texto)
    return [tuple(grupos[k]) for k in orden]


def cierres():
    d = _load().get("cierres")
    if isinstance(d, list) and d:
        return [(c.get("texto", ""), bool(c.get("nsfw"))) for c in d if c.get("texto")]
    return CIERRES_FABRICA


def alias():
    """Los de fábrica MÁS los tuyos: sumar no puede romper los que ya andaban."""
    propios = _load().get("alias")
    base = dict(ALIAS_FABRICA)
    if isinstance(propios, dict):
        base.update({_norm(k): v for k, v in propios.items() if k and v})
    return base


def guardar(cuerpos_=None, cierres_=None, alias_=None):
    """Guarda lo que le pases y deja lo demás como estaba.

    `cuerpos_` es la forma AGRUPADA: [(tags, [textos], nsfw)]. Se guarda como
    `textos` (array). Tolera que una variante venga como string suelto por si el
    llamador todavía manda la forma vieja."""
    d = _load()
    if cuerpos_ is not None:
        out = []
        for tags, textos, nsfw in cuerpos_:
            if isinstance(textos, str):
                textos = [textos]
            textos = [t.strip() for t in textos if (t or "").strip()]
            if tags and textos:
                out.append({"tags": tags, "textos": textos, "nsfw": bool(nsfw)})
        d["cuerpos"] = out
    if cierres_ is not None:
        d["cierres"] = [{"texto": x, "nsfw": n} for x, n in cierres_]
    if alias_ is not None:
        d["alias"] = alias_
    _save(d)
    return d


def exportar():
    """Los moldes ACTUALES como dict, agrupados por combinación (una fila con su
    array de variantes). Exporta lo efectivo (los de fábrica si nunca guardaste),
    así el archivo siempre sale completo y editable, no vacío."""
    return {
        "cuerpos": [{"tags": t, "textos": xs, "nsfw": n} for t, xs, n, _ in combinaciones()],
        "cierres": [{"texto": x, "nsfw": n} for x, n in cierres()],
        # solo TUS alias, no los de fábrica (esos ya los suma alias() solo). Si
        # exportáramos los 60 de fábrica, al reimportar quedarían clavados como
        # propios y "restaurar fábrica" no los sacaría.
        "alias": (_load().get("alias") or {}),
    }


def importar(data):
    """Reemplaza los moldes con un dict importado. Tolera basura: saca lo que no
    tenga la forma esperada en vez de romper. Devuelve cuántos cuerpos entraron.

    Levanta ValueError si el dict no tiene NADA usable, para que la web avise en
    vez de dejar los moldes vacíos sin que te des cuenta."""
    if not isinstance(data, dict):
        raise ValueError("el archivo no es un objeto JSON")
    cu = []
    for c in (data.get("cuerpos") or []):
        if not isinstance(c, dict):
            continue
        tags = [str(t).strip() for t in (c.get("tags") or []) if str(t).strip()]
        # acepta las dos formas: `textos` (array, nuevo) o `texto` (str, viejo)
        if isinstance(c.get("textos"), list):
            textos = [str(x).strip() for x in c["textos"] if str(x).strip()]
        else:
            uno = str(c.get("texto", "")).strip()
            textos = [uno] if uno else []
        if tags and textos:
            cu.append((tags, textos, bool(c.get("nsfw"))))
    ci = []
    for c in (data.get("cierres") or []):
        if isinstance(c, dict) and str(c.get("texto", "")).strip():
            ci.append((str(c["texto"]).strip(), bool(c.get("nsfw"))))
    al = {}
    if isinstance(data.get("alias"), dict):
        al = {str(k).strip(): str(v).strip() for k, v in data["alias"].items()
              if str(k).strip() and str(v).strip()}
    if not cu and not ci and not al:
        raise ValueError("no hay ni un molde válido en el archivo")
    guardar(cuerpos_=cu, cierres_=ci, alias_=al)
    return len(cu)


def _expandir(tags):
    """Las etiquetas de la foto, normalizadas y con los alias aplicados. Se
    conservan las dos formas (la original y la del alias) para que un molde
    escrito sobre cualquiera de las dos se prenda igual."""
    al = alias()
    out = set()
    for t in tags or []:
        n = _norm(t)
        if not n:
            continue
        out.add(n)
        if n in al:
            out.add(_norm(al[n]))
    return out


def _semilla(clave):
    """Número estable para esta foto: el mismo pie sale siempre igual, pero dos
    fotos distintas no comparten molde. Sin esto el pie cambiaría en cada
    recarga, que es peor que repetirse."""
    return int(hashlib.sha1(clave.encode("utf-8")).hexdigest()[:8], 16)


# Terminaciones de género que el comodín @ da por equivalentes. Cortas a
# propósito: así "pelirroj@" prende con pelirroja/pelirrojo/pelirroje pero el
# tronco no se come una palabra larga (pelirrojismo no entra).
_TERM_GENERO = {"", "a", "o", "e", "x", "as", "os", "es"}


def _coincide(tag_molde, presentes):
    """¿La etiqueta de un molde matchea alguna de las etiquetas de la foto?

    Sin @: coincidencia exacta (normalizada), como siempre.
    Con @: comodín de género. "pelirroj@" corta en el @ y prende con cualquier
    etiqueta que sea el tronco + una terminación de género (pelirroja, pelirrojo,
    pelirroje). Lo pone el usuario donde quiere, así nunca junta lo que no debe:
    "manzano" sin @ jamás matchea "manzana"."""
    raw = tag_molde or ""
    if "@" not in raw:
        n = _norm(raw)
        return bool(n) and n in presentes
    # _norm borra el @ (no es \\w), así que el tronco se saca ANTES de normalizar
    tronco = _norm(raw.split("@", 1)[0])
    if not tronco:
        return False
    return any(p.startswith(tronco) and p[len(tronco):] in _TERM_GENERO
               for p in presentes)


def candidatos(tags, sano=True):
    """Los cuerpos que se prenden con estas etiquetas, más específico primero.

    Se exigen TODAS las etiquetas del molde, no alguna. Con "alguna" pasaba esto:
    un molde de ["cama","desnuda"] se prendía en una foto etiquetada solo "cama"
    y el pie afirmaba que estaba desnuda cuando eso no estaba etiquetado. Los
    sinónimos y tipeos ya los cubre el mapa de alias, así que las listas de acá
    son requisitos, no alternativas.

    Gana el que exige más: ["cama","desnuda"] le gana a ["cama"] si están las
    dos, que es lo que querés."""
    presentes = _expandir(tags)
    out = []
    for disparo, texto, nsfw in cuerpos():
        if nsfw and sano:
            continue                      # sin adulto confirmado, no entra
        req = [d for d in disparo if (d or "").strip()]
        if req and all(_coincide(d, presentes) for d in req):
            out.append((len(req), texto, nsfw))
    out.sort(key=lambda x: -x[0])
    return out


def variantes(clave, tags, sano=True, maximo=12):
    """TODOS los pies posibles para esta foto, en un orden estable.

    Que sean varios es el punto: con uno solo, el botón de regenerar no servía
    para nada —devolvía siempre lo mismo—. El orden depende de la foto (por eso
    dos fotos con las mismas etiquetas no arrancan igual) pero no del reloj, así
    que recargar la página no te cambia el pie: solo pedir otro."""
    opciones = candidatos(tags, sano)
    if not opciones:
        return []
    # TODOS los que prendieron, no solo los más específicos. Quedándome con el
    # tramo de arriba salía casi siempre un único cuerpo y las "variantes" eran
    # la misma frase con distinto cierre: se notaba muchísimo. Los más
    # específicos van primero igual (candidatos() ya los ordena), así el mejor
    # sale de entrada y los generales quedan como alternativas al regenerar.
    s = _semilla(clave)
    mejor = opciones[0][0]
    top = [(t, n) for pega, t, n in opciones if pega == mejor]
    resto = [(t, n) for pega, t, n in opciones if pega != mejor]
    # rotar solo el tramo más específico: el primero depende de la foto, pero un
    # molde general nunca le gana a uno específico
    corte = s % len(top)
    empatados = top[corte:] + top[:corte] + resto

    out = []
    for j, (texto, cuerpo_nsfw) in enumerate(empatados):
        # Los cierres subidos SOLO con un cuerpo subido: si no, quedaba "Hecha
        # pomada pero contenta. Alguien va a tener que hacerse cargo." en una
        # foto de gimnasio.
        disponibles = [c for c, nsfw in cierres()
                       if not nsfw or (cuerpo_nsfw and not sano)]
        out.append(texto)                     # seco, sin cierre
        for k in range(2):                    # y con dos cierres distintos
            if disponibles:
                out.append(f"{texto} {disponibles[(s // 3 + j * 5 + k * 11) % len(disponibles)]}")
        if len(out) >= maximo:
            break
    vistos, fin = set(), []
    for v in out:
        if v not in vistos:
            vistos.add(v)
            fin.append(v)
    return fin[:maximo]


def escribir(clave, tags, sano=True, evitar=None):
    """El pie para esta foto, o None si ninguna etiqueta prendió un molde.

    `evitar` es el pie que ya tiene puesto: se devuelve el SIGUIENTE de la lista,
    en círculo. Así el botón de regenerar recorre las variantes en vez de
    devolverte siempre la misma, que era lo que pasaba.

    Devolver None es a propósito — el que llama decide si cae al modelo o deja el
    pie vacío."""
    vs = variantes(clave, tags, sano)
    if not vs:
        return None
    actual = (evitar or "").strip()
    if actual in vs:
        return vs[(vs.index(actual) + 1) % len(vs)]
    if actual:
        # tenía un pie que no salió de un molde (lo escribió el modelo, o lo
        # editaste a mano): se arranca por el primero
        return vs[0]
    return vs[0]


def cobertura(tags):
    """True si estas etiquetas prenden algún molde. Para que la web te muestre
    dónde te faltan moldes en vez de que lo descubras foto por foto."""
    return bool(candidatos(tags, sano=False))


# Etiquetas que NO son situación ni lugar: atributos y género/edad. No merecen
# molde (nadie postea "tengo el pelo castaño") y ensuciarían el ranking de
# "dónde faltan moldes". Se detectan por prefijo o por estar en esta lista.
_NO_DISPARAN = {
    "mujer", "hombre", "adulto", "adulta", "flaca", "gorda", "alta", "baja",
    "joven", "morocha", "rubia", "castaña", "pelirroja", "lacio", "suelto",
    "largo", "corto", "oscuro", "linda", "hermosa", "sola", "solo",
    "pendeja", "de frente", "de perfil",
}
_PREFIJOS_ATRIBUTO = ("pelo", "ojos", "ojoz", "piel", "tez")


def _es_atributo(tag):
    n = _norm(tag)
    return (n in _NO_DISPARAN
            or any(n.startswith(p) for p in _PREFIJOS_ATRIBUTO))


def huecos(conteo_por_tag):
    """De {etiqueta: en cuántas fotos aparece}, las que NO prenden ningún molde,
    ordenadas por las que más rinden. Sirven de sugerencia de dónde escribir.

    Se sacan los atributos (pelo, flaca, alta…): no van al pie a propósito, así
    que marcarlos como "hueco" sería ruido."""
    faltan = []
    for tag, n in conteo_por_tag.items():
        if _es_atributo(tag):
            continue
        if not candidatos([tag], sano=False):
            faltan.append((tag, n))
    faltan.sort(key=lambda x: (-x[1], x[0]))
    return faltan


# --- GENERADOR DE BORRADORES --------------------------------------------------
# El modelo escribe medio mal (por eso existen los moldes), así que esto NO da
# pies finales: da BORRADORES que el usuario después pule. La gracia es sembrar
# el trabajo — 300 combinaciones ya redactadas sobre TUS situaciones reales — en
# vez de una hoja en blanco. Se marca cada uno "borrador" hasta que lo edites.

def _situacionales(tags):
    """Las etiquetas de una foto que sirven de disparo: saca los atributos
    (pelo, flaca, ojos…) que no van al pie. Normaliza y saca repetidas."""
    out = []
    for t in tags or []:
        n = _norm(t)
        if n and not _es_atributo(t) and n not in out:
            out.append(n)
    return out


def recolectar_combos(fotos, k=3, max_tags=3):
    """Qué combinaciones vale la pena generar, con el umbral de frecuencia.

    `fotos` = [(tags, es_nsfw)] de la biblioteca. Devuelve [(tags, nsfw)]:
    - de 1 etiqueta: SIEMPRE (es el piso de cobertura).
    - de 2 y 3: solo si la combinación aparece en >= k fotos DEL MISMO registro
      (sano o crudo). Que se repita es la señal de que es una situación real y no
      un trío casual de etiquetas.

    Se cuenta sano y crudo por separado (la carpeta _nsfw decide el registro), así
    la misma combinación puede dar un molde sano Y uno crudo — las "dos versiones".
    """
    from collections import Counter
    from itertools import combinations
    cuenta = {False: Counter(), True: Counter()}   # por registro
    for tags, nsfw in fotos:
        sit = _situacionales(tags)
        vistos = set()
        for r in range(1, max_tags + 1):
            for combo in combinations(sorted(sit), r):
                if combo not in vistos:      # una foto cuenta 1 vez por combo
                    vistos.add(combo)
                    cuenta[bool(nsfw)][combo] += 1
    salida = []
    for nsfw, c in cuenta.items():
        for combo, n in c.items():
            if len(combo) == 1 or n >= k:
                salida.append((list(combo), nsfw, n))
    # más frecuentes primero, y las específicas antes que las de 1 sola
    salida.sort(key=lambda x: (-x[2], -len(x[0])))
    return [(tags, nsfw) for tags, nsfw, _ in salida]


_SYS_GEN_SANO = (
    "Escribís el pie de una foto tuya para redes, en PRIMERA persona, en español "
    "RIOPLATENSE (tenés, sos, mirá, acá; NUNCA tú, tienes, vale, tío). Te doy las "
    "cosas que hay en la foto y vos INVENTÁS la situación alrededor, como la "
    "contaría alguien que estuvo ahí. NO enumeres las etiquetas ni describas la "
    "foto: el que lee ya la ve. Una frase corta, natural, sin poesía ni moraleja.")
_SYS_GEN_CRUDO = _SYS_GEN_SANO + (
    " Es una foto subida de tono: escribí crudo y directo, las cosas por su nombre "
    "(tetas, culo, concha), con actitud, sin vergüenza y sin eufemismos.")


def generar_frases(tags, nsfw, n=3, model=None):
    """`n` pies BORRADOR para una combinación, pidiéndoselos al modelo (texto solo,
    sin imagen). Devuelve una lista de frases limpias; puede tener menos de `n` si
    el modelo repitió o devolvió basura. No levanta: ante error devuelve []."""
    reg = _SYS_GEN_CRUDO if nsfw else _SYS_GEN_SANO
    prompt = (f"En la foto hay: {', '.join(tags)}.\n"
              f"Escribí {n} pies DISTINTOS, uno por línea, sin numerarlos.")
    try:
        raw = vlm.generar_texto(prompt, model=model or vlm.DEFAULT_MODEL,
                                system=reg, num_predict=40 * n, temperature=0.95)
    except Exception:
        return []
    out = []
    for linea in raw.splitlines():
        t = re.sub(r"^\s*[-*•\d.)\"]+\s*", "", linea).strip().strip('"').strip()
        if 3 <= len(t) <= 160 and t not in out:
            out.append(t)
    return out[:n]


def fusionar_borradores(nuevos):
    """Suma moldes BORRADOR a los que ya hay, SIN pisar los tuyos. Cada `nuevo` es
    (tags, [frases], nsfw). Si ya existe un molde para esa combinación+registro,
    se le agregan las frases que falten; si no, se crea con borrador=True.

    Devuelve (combos_nuevos, frases_agregadas)."""
    d = _load()
    cuerpos_ = d.get("cuerpos")
    if not isinstance(cuerpos_, list):
        # partir de fábrica aplanada a la forma de disco, para no perderla
        cuerpos_ = [{"tags": t, "textos": [x], "nsfw": n} for t, x, n in CUERPOS_FABRICA]
    # índice por (combinación normalizada, nsfw)
    idx = {}
    for c in cuerpos_:
        clave = (tuple(sorted(_norm(t) for t in c.get("tags", []))), bool(c.get("nsfw")))
        idx[clave] = c
    combos_nuevos = frases_nuevas = 0
    for tags, frases, nsfw in nuevos:
        frases = [f.strip() for f in frases if (f or "").strip()]
        if not tags or not frases:
            continue
        clave = (tuple(sorted(_norm(t) for t in tags)), bool(nsfw))
        if clave in idx:
            c = idx[clave]
            tx = c.setdefault("textos", [c.pop("texto")] if c.get("texto") else [])
            for f in frases:
                if f not in tx:
                    tx.append(f)
                    frases_nuevas += 1
        else:
            nuevo = {"tags": tags, "textos": frases, "nsfw": bool(nsfw), "borrador": True}
            cuerpos_.append(nuevo)
            idx[clave] = nuevo
            combos_nuevos += 1
            frases_nuevas += len(frases)
    d["cuerpos"] = cuerpos_
    _save(d)
    return combos_nuevos, frases_nuevas
