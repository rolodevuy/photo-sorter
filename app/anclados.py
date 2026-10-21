"""Datos ANCLADOS de una foto, para que la narración no los adivine.

El paso 1 (Visión) escribe prosa mirando la foto, y ahí está su punto débil: el
VLM de 7B, cuando redacta un párrafo largo sobre un cuerpo, rellena con lo que el
contexto sugiere. Medido: en `acro_0008` (donde hay bragas grises) escribió
"pantalones bajados a las nalgas"; en `acro_0015` (donde la visión dice solo
"lencería negra") inventó "el centro desgarrado para exponer su huevo". El acto y
la piel son justo lo que peor ve y lo que más inventa.

La app ya tiene tres detectores dedicados que SÍ son confiables, cada uno
entrenado para lo suyo:

  - ACTO (sexpos): resnet50 porntech/sex-position. Qué están haciendo.
  - PIEL/GENITALIA (nsfw.py, NudeNet): qué partes están CUBIERTAS por la ropa.
    Es el dato que evita el invento: da "genitales cubiertos" donde el VLM
    escribía "desnuda". No se usa para afirmar desnudo (ver _PARTES_TAPADO).
  - QUIÉN (finder.who_is_in): reconocimiento facial contra los conocidos.

Esto junta los tres en un texto corto y en jerga, que se le pasa al VLM como
DATO, no como sugerencia: el modelo lo usa y no lo discute.

Lo que NO hace y por qué:
  - El acto corre solo en foto NSFW y con adulto confirmado. Es la misma
    compuerta que ya usan el pie y los comentarios (sexpos no tiene sentido en
    una foto normal: tira cualquier cosa, lo avisa su propio autor).
  - NudeNet corre siempre: es un detector de partes del cuerpo, no de actos, y
    en una foto vestida dice "tapada", que es exactamente el dato que evita el
    invento de desnudo.
  - El reconocimiento facial puede no encontrar nada (foto sin cara: un torso,
    una espalda). Ahí no hay nombre y el texto no nombra.

Todo local, todo en try/except: ningún detector puede romper la narración.
"""

from pathlib import Path

from app import NSFW_DIR_NAME

# NudeNet: solo lo que sirve para narrar. Las caras se descartan (ya las usa el
# reconocimiento) y "belly/feet" no aportan a lo que se cuenta.
#
# SOLO LO TAPADO. Parece raro, pero es la decisión central de este módulo y está
# medida. NudeNet acierta el estado de la ropa mucho mejor que el VLM, pero
# AFIRMA DESNUDO CUANDO NO LO HAY: en tizi_0000 (una foto vestida) dijo
# "FEMALE_BREAST_EXPOSED" con 0.836, un score que ningún umbral filtra — es un
# error del detector, no de calibración. Y en acro_0008 (bragas grises puestas)
# tiró "GENITALIA_COVERED 0.412" junto a "BREAST_EXPOSED 0.405": los dos estados
# del mismo cuerpo con 0.007 de diferencia.
#
# Pasarle eso a la narración como dato duro sería darle permiso para inventar
# desnudo. Así que de NudeNet solo se usa el lado conservador: "esto está tapado
# por la ropa". Eso nunca agrega desnudo, solo lo niega — y negarlo es lo que
# frena el invento.
#
# QUIÉN AFIRMA EL DESNUDO: el VLM mirando la foto (decisión explícita del
# usuario). Este módulo es el FRENO, no el juez: si dice "cubierto", la pasada 2
# lo respeta; si la descripción dice "desnuda" y acá no hay nada que la
# contradiga, se describe desnuda. Es a sabiendas, porque el detector solo no
# sirve (ver arriba: tizi_0000, vestida, BREAST_EXPOSED 0.836).
_PARTES_TAPADO = {
    "FEMALE_GENITALIA_COVERED": "genitales cubiertos por la ropa",
    "FEMALE_BREAST_COVERED": "pechos cubiertos por la ropa",
    "BUTTOCKS_COVERED": "nalgas cubiertas por la ropa",
    "ANUS_COVERED": "ano cubierto por la ropa",
}

SCORE_PARTES = 0.35     # umbral de NudeNet para el dato. Más alto que el del
                        # revisor NSFW (0.45 no, 0.30 tampoco): a 0.30 entraba el
                        # "genitales tapados 0.344" de acro_0008 como si fuera
                        # firme. A 0.35 queda solo lo que el detector sostiene.
SCORE_ACTO = 0.45       # el mismo de sexpos

_det = None


def _nudenet():
    """NudeNet perezoso (el import tarda). None si no está instalado."""
    global _det
    if _det is None:
        try:
            from nudenet import NudeDetector
            _det = NudeDetector()
        except Exception:
            _det = False
    return _det or None


# El acto que da el clasificador es una CATEGORÍA, no una escena: dice "chupando"
# sin decir a quién ni con qué. Medido: pasándolo así, el redactor escribe
# "chupa algo" — tiene el dato y no sabe qué hacer con él. Acá se expande a la
# escena concreta y soez, UNA sola vez, en el DATO (no en el system, que se
# devuelve como eco; ni en el redactor, que si no inventa).
#
# El detector NO distingue misionero/doggy/anal ni a quién: "hardcore" es
# penetración genérica (lo avisa su propio autor). Así que la escena se arma con
# lo que la categoría sí afirma, sin agregar una posición ni un agujero que el
# detector no vio. "solo" no lleva escena: no hay acto con nadie.
_ACTO_ESCENA = {
    "chupando": "le está chupando la pija a alguien",
    "penetración": "la están penetrando",
    "rusa": "le está haciendo una rusa con las tetas",
    "paja": "le está haciendo una paja a alguien",
    "comiéndole la concha": "le está comiendo la concha a alguien",
    "dedeando": "se está dedeando o le está dedeando a alguien",
    "sola": "",                     # sin acto con otro: no hay escena que armar
}


def _acto(image_path, es_nsfw, sano):
    """El acto, en jerga, o None. Solo en NSFW y con adulto confirmado."""
    if not es_nsfw or sano:
        return None
    try:
        from app import sexpos
        if not sexpos.disponible():
            return None
        r = sexpos.clasificar(str(image_path))
    except Exception:
        return None
    return r[0] if r else None


def _partes(image_path):
    """Lo que NudeNet ve CUBIERTO por la ropa, sin repetir.

    Solo "cubierto" a propósito: ver _PARTES_TAPADO. El detector no se usa para
    afirmar desnudo (se equivoca: tizi_0000 a 0.836), solo para negarlo — que es
    el dato que evita que la narración invente."""
    det = _nudenet()
    if not det:
        return []
    try:
        dets = det.detect(str(image_path))
    except Exception:
        return []
    vistas = []
    for d in dets:
        if d["score"] < SCORE_PARTES:
            continue
        txt = _PARTES_TAPADO.get(d["class"])
        if txt and txt not in vistas:
            vistas.append(txt)
    return vistas


def _quienes(image_path, duenio):
    """Los nombres reconocidos por cara, con la dueña de la carpeta primero.

    El reconocimiento es cara por cara y exige margen contra el segundo
    candidato (ver finder.who_is_in): si duda entre dos personas no nombra
    ninguna, que es lo correcto para algo que se guarda como dato de la foto.

    Devuelve (otros, ok, duenio_en_foto). `duenio_en_foto` es True solo si la
    CARA de la dueña se reconoció de verdad (ver faces_in) — no si se asumió por
    estar la foto en su carpeta. Sirve para el sujeto del acto: "chupando" es una
    categoría sin sujeto, y con la cara reconocida se puede afirmar de quién es.
    """
    try:
        from app import finder
        caras = finder.faces_in(str(image_path))
    except Exception:
        return [], False, False
    nombres = [c["name"] for c in caras]
    otros = [n for n in nombres if n != duenio]
    return otros, True, duenio in nombres


def _vinculo_linea(tags, duenio):
    """La relación con el dueño, en UNA línea, o "".

    El vínculo ya existe en el sistema de perfiles (moldes.relacion_de: lo que
    ELLA es para vos — prima, ex, amiga) y se usa en el pie. La Visión no lo
    miraba, y eso deja al narrador tratando a una ex igual que a una desconocida.
    Va como dato para que la trate según el vínculo, no para que lo recite.

    `relacion_para_duenio` da el vínculo DADO VUELTA (prima -> primo, cómo te
    llama ella a vos): es lo que necesita un narrador en primera persona que le
    habla a ella ("vení, primita").
    """
    if not tags:
        return ""
    try:
        from app import moldes
        vinc = moldes.relacion_de(tags)
    except Exception:
        return ""
    if not vinc:
        return ""
    try:
        rec = moldes.relacion_para_duenio(vinc) or ""
    except Exception:
        rec = ""
    txt = ("Vínculo con %s: ella es tu %s" % (duenio, vinc))
    if rec and rec != vinc:
        txt += "; ella te llama %s" % rec
    txt += (". Ese vínculo te calienta MÁS, no menos: si es tu amiga, que sea tu "
            "amiga te la pone más dura; si es tu ex, saber que es tu ex te dan "
            "más ganas de cogerla. Usalo como morbo en el deseo — decí que te "
            "excita justamente eso — sin convertirlo en el tema y sin recitarlo "
            "como etiqueta.")
    return txt


def anclados(image_path, duenio, rel="", es_nsfw=False, sano=True, tags=None):
    """El bloque de datos duros de la foto, listo para meter en el prompt.

    Devuelve (texto, nombres) donde `nombres` es la lista de gente reconocida
    (vacía si no hay). El texto va en español y en jerga, corto, porque va como
    DATO del prompt, no como narración.

    duenio: nombre de la carpeta (el perfil). Si no se reconoce a NADIE MÁS, se
    asume que la foto es suya y se la nombra — con cara o sin ella (decisión del
    usuario: en una carpeta de persona, la mayoría de las fotos son de ella; el
    costo es que una foto de una amiga sin reconocer también sale con su nombre,
    y es a sabiendas).
    tags: las etiquetas del perfil (tags.get_person). De ahí sale el vínculo;
    si no vienen, la narración simplemente no lo menciona."""
    image_path = Path(image_path)
    acto = _acto(image_path, es_nsfw, sano)
    partes = _partes(image_path)
    otros, ok, duenio_en_foto = _quienes(image_path, duenio)

    lineas = []
    # El acto NO va como un dato neutro ("Acto detectado: chupando"): el
    # clasificador da una CATEGORÍA, y así el redactor escribe "chupa algo"
    # (medido). Va la ESCENA concreta (ver _ACTO_ESCENA) y como lo que es — un
    # hecho de la foto confirmado por un detector que la miró, distinto del
    # deseo del narrador. "sola" no arma escena: no hay acto con otro.
    #
    # EL SUJETO: la categoría viene sin sujeto ("chupando" no dice de quién). Si
    # la cara reconocida ES la dueña (duenio_en_foto, de faces_in: reconocida de
    # verdad, no asumida por carpeta), se puede afirmar de quién es el acto. Antes
    # el modelo no lo sabía y escribía "chupando" a secas; ahora dice de quién.
    escena = _ACTO_ESCENA.get(acto or "", "")
    if acto and escena and duenio_en_foto:
        escena = "%s es quien lo hace: %s" % (duenio, escena)
    if acto and escena:
        lineas.append(
            "Hecho confirmado de la foto (lo detectó un clasificador que la "
            "miró, no es un deseo): %s. Narrá ESE acto, concreto y soez, en la "
            "descripción: no lo cites, no hables del clasificador, no lo "
            "suavices ni digas \"algo\"." % escena)
    elif acto == "sola":
        lineas.append("Hecho confirmado de la foto: está sola, no hay nadie "
                      "más con ella.")
    if partes:
        lineas.append("Cuerpo detectado por clasificador: %s." % ", ".join(partes))
    vinc = _vinculo_linea(tags, duenio)
    if vinc:
        lineas.append(vinc)
    nombres = []
    if otros:
        nombres = [duenio] + [n for n in otros if n != duenio]
    elif ok:
        # Se asume la dueña de la carpeta aunque HAYA cara: la foto está en SU
        # carpeta, y lo normal es que sea ella. Antes esto exigía `not hay_cara`
        # y eso dejaba afuera justo el caso MÁS común — una foto con cara que el
        # reconocimiento no pudo matchear (ángulo, NSFW, baja calidad). Medido:
        # acro_0015 tiene cara y no se reconoció, así que no se nombraba y
        # salía "la mujer de pelo rojo". Decisión del usuario: nombrar siempre.
        # El costo es el mismo de antes (una foto de una amiga sin reconocer
        # también sale con su nombre), y es a sabiendas.
        nombres = [duenio]
    if nombres:
        if len(nombres) == 1:
            lineas.append("Persona reconocida en la foto: %s." % nombres[0])
        else:
            lineas.append("Personas reconocidas en la foto: %s."
                          % ", ".join(nombres))
    return "\n".join(lineas), nombres


def nombre_para_narrar(nombres, duenio):
    """Cómo nombrar en la narración, o "" si conviene no nombrar.

    Con una sola persona se nombra. Con varias se nombra solo si la dueña está
    (es su foto): se la nombra a ella y se dice que hay más gente, sin arriesgar
    quién es quién."""
    if not nombres:
        return ""
    if len(nombres) == 1:
        return nombres[0]
    if duenio in nombres:
        return duenio
    return ""
