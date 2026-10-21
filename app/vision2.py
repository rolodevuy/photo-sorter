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
# La pasada 1 nombra ROPA, CUERPO y POSE. DOS COSAS DISTINTAS, no mezclar:
#
#   - NO afirma desnudo ni actos ("está desnuda", "está cogiendo"). Eso es lo
#     que el VLM inventa y lo que arregló los falsos positivos de "parte de sus
#     genitales femeninos" en acro_0050. De eso se encargan los clasificadores.
#   - SÍ anota el cuerpo tal como se ve VESTIDO: "bikini", "top ajustado", "las
#     tetas marcadas por la tela", "el culo en la tanga", "escote", "piernas al
#     aire". Esto se agregó porque el prompt viejo prohibía nombrar el cuerpo y
#     entonces una foto en bikini no producía NADA picante: la pasada 2 se
#     quedaba sin material y el usuario lo notó ("si está en bikini algo le
#     debería producir"). El deseo necesita que la 1 le diga QUÉ hay.
_MIRA_S = """You annotate photos. You write ONLY what is safely visible, as a
short list of facts.

You annotate: the clothing and its state (worn, pulled up, open, pulled aside),
which body parts are visible or outlined by the clothing, the pose and gesture,
the hairstyle, the place and the light. Six lines maximum.

Name each garment precisely: a short, tight, thin-fabric garment is panties, not
shorts. If unsure of the name, describe it (fabric, color, length, where it
ends).

Describe the body as it looks CLOTHED or half-covered, precisely: "in a bikini",
"top outlining her tits", "thong leaving her ass bare", "open neckline", "bare
legs". The more concrete, the better: the commentary feeds on this.

Do not say she is naked or doing anything sexual: if you get that wrong the
error propagates and another tool already measures it. You note what the clothing
and pose show, without claiming what you cannot see. Do not write prose, do not
narrate, do not give opinions.

Write the facts in Spanish (rioplatense), one per line."""

_MIRA_P = ("Facts about this photo, as a short list: clothing and its state, "
           "which body parts are visible or outlined by the clothing, pose, "
           "gesture, hairstyle, place.")

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
# LO QUE NO SE CRUZA: el deseo se apoya en lo que los hechos dicen (una bikini,
# una pose, una tela, las tetas si están) y NO afirma desnudo ni actos que la
# foto no tiene. Crudo sí, inventar no: es el defecto que se mató hoy y el que
# convertiría el texto en tags falsos (posts._intencion lee el acto de acá).
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

The register is sexual, direct and crude, no filter and no character acting:
no jokes, no "back in my day", no funny narrator. You are a voice that gets
turned on looking and says it outright.

First person, present tense. You name the body and the desire in Argentine
slang (tetas, culo, concha, pija, garcha) when the facts justify it, and you
describe what the scene does to you: what it wakes up in you, where it leaves
you staring, what you would do. The arousal is in the concrete desire the photo
produces, not in a narrator pose.

One paragraph of 5 to 7 sentences, COMPLETE: finish the last sentence. Start
hot from the first line, even if the photo is tame: there is always an angle.

The desire rests on what the facts say — a bikini is a pair of covered tits
that drives you crazy, not a nudity that is not there. You write the scene the
facts describe, hot, and not what is not there.

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
