"""La «visión» de una foto en DOS PASADAS, porque una sola inventa.

POR QUÉ DOS Y NO UNA (esto está medido, no es teoría):

Pedirle a un VLM que mire la foto y además redacte un párrafo largo es pedirle
dos cosas que se pelean. Cuando redacta, rellena. Con el 7B se probaron SEIS
arquitecturas (crudo mirando, crudo sobre texto, neutral largo, neutral corto,
datos anclados en el prompt, y dos pasadas) y todas inventaban desnudo donde no
lo había. En acro_0008 (bragas grises puestas) salió "pantalones bajados a las
nalgas"; en tizi_0000 (vestida) salió "torso desnudo". Los datos de los
clasificadores los ignoraba: le inyecté "genitales cubiertos por la ropa" y
escribió "expuesto" igual.

La división que SÍ funciona, probada sobre fotos de acro:

  1) MIRAR (7B, mira la foto): no redacta, solo anota HECHOS CORTOS — prendas y
     su estado, qué partes del cuerpo se ven o quedan marcadas por la ropa,
     pose, gesto, peinado, lugar. Y no afirma desnudo ni actos: de eso se
     encarga el clasificador. Pidiendo corto y sin afirmar desnudo, el 7B deja
     de inventar (fue el cambio que arregló los falsos "genitales").
  2) REDACTAR (9B abliterated, NO mira la foto): escribe el comentario desde los
     hechos. Un redactor dedicado respeta un límite que el VLM mirando no.

El dato anclado (acto, ropa cubierta, nombre) va en la pasada 2, que es la que lo
respeta. En la 1 se pierde.

EL REGISTRO es un pedido del usuario tras TRES iteraciones y no una decisión
técnica: quiere crudeza sexual directa, en jerga y siempre subida, sin voz de
personaje. Se probó antes con prosa literaria de fototeca (rechazada: mucha
lírica) y con un tío borracho aleccionando (rechazada: "suena a chiste"). El
deseo es explícito; el cuerpo sigue saliendo de los hechos (ver _RED_S).

LOS SYSTEM VAN EN INGLES (pedido del usuario, hipótesis: los modelos están
entrenados sobre todo en inglés y obedecen mejor). La SALIDA se pide en español
en el propio system, así que no hay traductor: se descartó a propósito porque
sumaría un tercer paso lento y porque una mala traducción se convertiría en tag
falso. Si el texto sale en inglés igual, reforzar la línea "Write ... in
Spanish", que es la que lo sostiene.

COSTO: el redactor era el 30B (18 GB, ~130s por foto, medido). Se cambió al 9B
(6.6 GB) para bajar el tiempo. Sigue siendo lento en CPU, así que esto NO corre
en tandas automáticas: es un botón, una foto, y queda guardado.
"""

from app import anclados, vlm

# El que MIRA. Es el VLM que ya usa la app para el pie y las etiquetas.
MIRA = "hf.co/mradermacher/Qwen2.5-VL-7B-NSFW-Caption-V3-abliterated-GGUF:Q4_K_M"

# El que REDACTA. Solo texto, sin foto: por eso no puede inventar mirando.
#
# CAMBIADO del 30B al 9B por PESO, no por calidad: el 30B son 18 GB y tardaba
# ~130s por foto (medido), lo que hacía impracticable el botón. El 9B es 6.6 GB
# y es abliterated (sin los rechazos que romperían un cuento subido de tono).
# No se midió todavía contra el 30B: si el texto sale flojo, el 30B sigue
# estando y se cambia esta línea sola.
REDACTOR = "huihui_ai/qwen3.5-abliterated:9b"

# Los system van en INGLES A PROPOSITO: los modelos estan entrenados sobre todo
# en ingles y obedecen mejor una instruccion fina ("bragas, no pantalon corto")
# escrita en ingles. Pero la SALIDA se pide en espanol desde el propio system, en
# una linea explicita: asi no hace falta un traductor despues (que sumaria un
# tercer paso lento y podria lavar la jerga o meter un error que se vuelve tag
# falso). Si el texto sale en ingles igual, la linea "Write ... in Spanish" es
# lo primero a reforzar.
#
# La pasada 1 nombra ROPA, CUERPO, DESNUDO y POSE. Esto cambio DOS VECES, y cada
# cambio tiene su porqué medido:
#
#   - V1 no dejaba nombrar el cuerpo ("no nombres partes íntimas") y su efecto
#     lateral era que una foto en bikini no producía NADA picante: la pasada 2 se
#     quedaba sin material ("si está en bikini algo le debería producir").
#   - V2 dejaba nombrar el cuerpo VESTIDO pero SEGUÍA prohibiendo decir "desnuda"
#     ("do not say she is naked"). Efecto medido: el usuario leía solo el deseo
#     del narrador y NO la descripción ("no me dice 'está frente al espejo,
#     desnuda, concha prominente y mojada'"). La descripción faltaba porque la 1
#     no la traía.
#   - V3 (la de ahora): la 1 SÍ describe lo que ve, INCLUIDO el desnudo, cuando
#     lo ve. Se le pide nombrar el cuerpo desnudo con precisión (concha, tetas,
#     areolas, culo) si está a la vista.
#
# QUIÉN DECIDE QUE HAY DESNUDO (decisión explícita del usuario, es la de riesgo):
# lo decide el VLM MIRANDO, y el detector (NudeNet) queda como FRENO. Es lo
# contrario de lo que se hizo todo el día, y se elige a sabiendas: el detector
# dedicado da falsos positivos de desnudo (tizi_0000 vestida: BREAST_EXPOSED
# 0.836, ningún umbral lo filtra), así que no sirve como único juez. El freno
# funciona al revés: si el detector dice "cubierto", la pasada 2 tiene el dato y
# lo respeta; si la 1 dice "desnuda" y el detector no la contradice, se describe
# desnuda. Riesgo aceptado: si el VLM ve mal, describe desnudo donde no hay.
#
# Lo que SIGUE prohibido en la 1: afirmar ACTOS ("está cogiendo"). Los actos no
# son descripción y son el dato que alimenta los tags NSFW (posts._intencion).
_MIRA_S = """You annotate photos. You write ONLY what is safely visible, as a
short list of facts.

You annotate: the clothing and its state (worn, pulled up, open, pulled aside),
which body parts are visible — clothed, outlined by the clothing, or bare —, the
pose and gesture, the hairstyle, the place and the light. Six lines maximum.

Name each garment precisely: a short, tight, thin-fabric garment is panties, not
shorts. If unsure of the name, describe it (fabric, color, length, where it
ends).

Describe the body precisely, in Argentine slang, as it actually looks: "in a
bikini", "top outlining her tits", "thong leaving her ass bare", and when it is
really uncovered, say it plainly — "tits out, nipples visible", "pussy on
display, shaved", "bare ass up in the air". Do not soften or euphemize: if the
body is there, name it. The commentary feeds on this.

Do not say she is doing anything sexual: if you get that wrong the error
propagates and another tool already measures it. Do not write prose, do not
narrate, do not give opinions.

Write the facts in Spanish (rioplatense), one per line."""

_MIRA_P = ("Facts about this photo, as a short list: clothing and its state, "
           "which body parts are visible — clothed or bare —, pose, gesture, "
           "hairstyle, place.")

# La pasada 2 escribe la ficha. Las reglas van en el `system` (en el prompt el
# modelo las devuelve como eco: visto el narrador citando su propia instrucción).
#
# LA VOZ (pedido del usuario, 3 iteraciones): lo que se quiere es CRUDEZA SEXUAL
# DIRECTA, no una voz. Vale la pena leer las tres porque cada corrección tiene su
# porqué:
#
#   1) Prosa literaria de fototeca ("cortina de seda", "el tiempo detenido").
#      RECHAZADA: mucha lírica, poco erotismo.
#   2) Tío borracho chapado a la antigua, con "antes las mujeres...", "en mi
#      época...", aleccionando. RECHAZADA: "suena a chiste, parece un diálogo de
#      un cuento del año 30". El problema fue pedir un PERSONAJE cómico: el
#      modelo actúa el sketch en vez de calentarse. También salió sin nada xxx:
#      el propio prompt decía "el morbo está en cómo mirás, no en lo que le
#      agregás", o sea que le PROHIBÍA la carga sexual.
#   3) La de ahora: deseo concreto y explícito, sin personaje, en jerga, y
#      siempre subido aunque la foto no dé (el nivel no se adapta: se fuerza).
#
# LA DESCRIPCION QUE FALTABA (quinta corrección, del usuario): el texto tenía
# deseo y nada más. "ahora leo solo lo que quiere el narrador, pero no me dice
# 'está frente al espejo, desnuda, concha prominente y mojada, tetas perfectas
# con areolas hermosas'". Al subir el tono, el modelo se gastaba todo el párrafo
# en el deseo y la descripción quedaba afuera. La causa era pedir una sola cosa
# ("what you would do"): el deseo se comía el espacio.
#
# El arreglo es estructural y por eso se escribe como DOS MITADES obligatorias:
# primero QUÉ HAY (descripción explícita de la foto, 2-3 frases), después QUÉ
# DESEA (3-4 frases de acto concreto). Así ninguna de las dos se come a la otra.
#
# Y la regla que hace las dos verdad: la descripción sale de los HECHOS (si dice
# desnuda, desnuda; si dice vestida, el cuerpo vestido), y el desnudo que no está
# va en la mitad del DESEO, dicho como deseo ("me imagino la concha que debe
# haber debajo"), nunca como si la foto lo mostrara.
#
# EL REPERTORIO SOEZ (sexta corrección, del usuario): "aun me faltan conceptos
# como meterle un pijazo, llenarla de lechita y cosas obscenas". La causa era que
# la mitad 2 nombraba el acto con UNA lista corta de verbos, y el modelo se
# quedaba con esos y no ampliaba. Se pidió el repertorio ENTERO por categorías
# (cómo la coge, dónde acaba, cómo la deja, qué le dice), siempre dirigido a ella
# y como DESEO. NO se enumeró un vocabulario nuevo: ya está medido que una lista
# larga de palabras en el prompt se devuelve como eco. El usuario eligió
# explícitamente que el acto NO se describa como si pasara en la foto (eso rompe
# los tags), solo como lo que él le haría.
#
# LO QUE NO SE CRUZA: el deseo se apoya en lo que los hechos dicen (una bikini,
# una pose, una tela, las tetas si están) y NO afirma desnudo ni actos que la
# foto no tiene. Crudo sí, inventar no: es el defecto que se mató hoy y el que
# convertiría el texto en tags falsos (posts._intencion lee el acto de acá).
#
# LA LINEA FINA DEL ACTO (cuarta corrección, del usuario): dijo que el texto
# tenia carga pero con EUFEMISMO — "me la traeria conmigo y le haria cosas" en
# vez de "meterle la pija es lo unico que me interesa de ella". El problema no
# era el vocabulario (la jerga estaba) sino que el modelo ESQUIVABA EL VERBO:
# decia que haria en abstracto en vez de nombrar el acto concreto. Y eso lo pedia
# el prompt sin querer, porque decia "what you would do" (que se contesta con un
# eufemismo).
#
# La distincion que resuelve las dos cosas a la vez, y que hay que mantener:
#   - el acto como DESEO del que mira ("le meteria la pija hasta que...") es
#     legítimo y es lo que se pide: es lo que la foto le produce.
#   - el acto como HECHO de la foto ("le estan metiendo la pija") es invento, y
#     rompe los tags. Eso no.
# Verbo concreto SI, afirmar que pasa en la foto NO.
#
# DOS MEDICIONES SOBRE ESTE SYSTEM, no volver atrás sin leerlas:
#
#   1) El DETALLE FINO. Con la primera versión el redactor ya no inventaba
#      desnudo, pero ADORNABA: sobre un hecho que decía "bra negro con las manos
#      cerca del pecho" escribió "el pulgar presiona directamente contra el
#      pezón" y "una mano se apoya sobre los glúteos".
#
#   2) El ECO. Cuando la lista de prohibiciones fue larga y enumerada, el modelo
#      empezó a CONTESTARLA dentro del texto: acro_0050 terminó con "sin añadir
#      ni un solo detalle más allí donde no estaba presente en los hechos dados".
#      Cuanto más enumerada la regla, más la devuelve. Por eso el system dice el
#      límite en general y no enumera.
_RED_S = """You write the commentary for a photo. You do not see the photo:
you only have the FACTS, and the facts are ALL you know.

The commentary has TWO HALVES, in this order, and both are required. First you
DESCRIBE what is in the photo, then you DESIRE her out loud. Neither half alone
is the job: describing without wanting is a dry inventory, and wanting without
describing leaves the reader without the photo.

HALF ONE — THE PHOTO (2 or 3 sentences). You describe, precisely and
explicitly, what the facts say is there: the pose, the place, and the body —
what is clothed, what is bare. If the facts say she is naked, you say she is
naked and you name what is on display (tetas, areolas, concha, culo) with the
words of the street. If the facts say she is dressed, you describe the body as
the clothing shows it, with the same hunger. This half is the photo, not your
wish: you write what the facts state, nothing more.

HALF TWO — THE DESIRE (3 or 4 sentences). Now you go for it, first person,
present, crude, no filter, no character acting. This is where the whole filthy
repertoire comes out, and it is all aimed AT HER: how you would fuck her, where
you would cum, how you would leave her, what you would say to her while you do
it. Not one act described once and done — go through it: how you take her, how
hard, where she ends up, what you pour on her, how she looks wrecked afterwards.
Meterle un pijazo, llenarla de lechita, dejarla temblando, hacerla gemir,
decirle todas las guarradas. If it is obscene, it belongs here.

That act is YOUR DESIRE looking at her, never something happening in the photo:
you say what you would do, never that it is being done. When the facts do not
give you a naked body, the desire is where the naked body goes — "me imagino la
concha que debe haber debajo" — said as a wish, never as something the photo
shows.

Do not soften and do not hint: the euphemism ("me la llevaría", "le haría
cosas", "la pasaría bien") is the failure to avoid. If a sentence could appear
in a magazine, it is too tame.

One paragraph, 5 to 7 sentences total, COMPLETE: finish the last sentence. The
register is on from the first line.

Write the commentary in Spanish (rioplatense), never in English. The slang must
sound Argentine, not translated: escribí como habla un argentino, no como una
traducción."""


def _hechos(image_path, model=None):
    """Pasada 1: mira la foto y anota hechos cortos. Sin redactar."""
    return (vlm.describe(str(image_path), model=model or MIRA, prompt=_MIRA_P,
                         system=_MIRA_S, num_predict=200,
                         temperature=0.3) or "").strip()


def _ficha(hechos, anclas="", nombre="", model=None, system=None):
    """Pasada 2: redacta desde los hechos. No mira la foto (generar_texto no
    manda imagen: es lo que garantiza que no agregue lo que no está).

    El andamiaje del prompt va en INGLES (igual que el system: el modelo
    obedece mejor) pero el texto de salida se pide en espanol desde _RED_S. Los
    HECHOS llegan en espanol de la pasada 1 y los anclados tambien: el modelo
    mezcla sin problema y es lo que se busca.

    El dato anclado se presenta como un hecho MÁS, sin imperativos ("esto
    manda", "respetá esto"): se midió que el modelo contesta el imperativo
    dentro de la ficha ("no hay rastro de otros cuerpos"). Un dato más en la
    lista, en cambio, lo escribe sin comentarlo — y como el anclado del cuerpo
    dice el estado real ("cubiertos por la ropa"), corrige al VLM sin prohibir
    nada.

    `system` permite cambiar el REGISTRO sin tocar la mecánica: el default es
    _RED_S (xxx crudo). Cambia la voz, no los hechos."""
    pr = "FACTS about the photo:\n\n%s\n\n" % hechos
    if anclas:
        pr += "Data from the classifiers:\n%s\n\n" % anclas
    if nombre:
        pr += "The person is %s.\n\n" % nombre
    pr += "Write the commentary now, as one full paragraph, in Spanish."
    return (vlm.generar_texto(pr, model=model or REDACTOR,
                              system=system or _RED_S,
                              num_predict=500, temperature=0.45) or "").strip()


def narrar(image_path, duenio, rel="", es_nsfw=False, sano=True,
           model_mira=None, model_redactor=None, log=None):
    """La ficha de la foto, en dos pasadas. Devuelve el texto final.

    image_path: la foto.
    duenio: el nombre de la carpeta (para el reconocimiento facial).
    es_nsfw / sano: la compuerta del acto — no corre con gente sin confirmar
    como adulta, igual que el pie y los comentarios.
    log: opcional, función(str) para ir contando qué etapa va.

    Devuelve "" si el modelo no devolvió nada (el llamador decide qué hacer).
    """
    dec = log or (lambda _s: None)
    dec("mirando la foto…")
    h = _hechos(image_path, model=model_mira)
    if not h:
        return ""
    anclas, nombres = "", []
    try:
        anclas, nombres = anclados.anclados(image_path, duenio, rel=rel,
                                            es_nsfw=es_nsfw, sano=sano)
    except Exception:
        pass                        # los detectores nunca rompen la narración
    nom = anclados.nombre_para_narrar(nombres, duenio)
    dec("redactando…")
    return _ficha(h, anclas=anclas, nombre=nom, model=model_redactor)
