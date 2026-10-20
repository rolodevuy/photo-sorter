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
     su estado, pose, gesto, peinado, lugar. Y no opina sobre el cuerpo desnudo:
     de eso se encarga el clasificador. Pidiendo corto y sin el cuerpo, el 7B
     deja de inventar (fue el cambio que arregló los falsos "genitales").
  2) REDACTAR (9B abliterated, NO mira la foto): escribe el comentario desde los
     hechos. Un redactor dedicado respeta un límite que el VLM mirando no.

El dato anclado (acto, ropa cubierta, nombre) va en la pasada 2, que es la que lo
respeta. En la 1 se pierde.

EL REGISTRO es un pedido del usuario y no una decisión técnica: la voz es de un
tío borracho, chapado a la antigua, mirando a una mujer que le gusta. Se probó
antes con prosa literaria de fototeca y se rechazó por lírica. La opinión es el
tono; el cuerpo sigue saliendo de los hechos (ver _RED_S).

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

# La pasada 1 nombra solo ROPA y POSE. No opina sobre el cuerpo desnudo: eso lo
# dicen los clasificadores. Este es el cambio que arregló los falsos positivos
# de "parte de sus genitales femeninos" en acro_0050.
_MIRA_S = """Sos un anotador de fotos. Anotás SOLO lo que se ve con seguridad, en
una lista corta de hechos.

Anotás: qué prendas lleva y en qué estado (puesta, subida, abierta, corrida),
la pose y el gesto, el peinado, el lugar y la luz. Máximo 5 líneas.

Nombrados cada prenda con la palabra justa: una prenda corta ajustada de tela
fina son bragas, no un pantalón corto. Si dudás del nombre, describila (tela,
color, largo, dónde llega).

No nombrés partes íntimas del cuerpo ni digas si algo está desnudo o expuesto:
de eso se encarga otra herramienta y si te equivocás el error se propaga. No
redactás, no describís, no opinás."""

_MIRA_P = ("Hechos de la foto, en lista corta: prendas y su estado, pose, "
           "gesto, peinado, lugar.")

# La pasada 2 escribe la ficha. Las reglas van en el `system` (en el prompt el
# modelo las devuelve como eco: visto el narrador citando su propia instrucción).
#
# LA VOZ (pedido del usuario): tío borracho, chapado a la antigua, mirando una
# mujer que le gusta. Antes era prosa literaria de fototeca ("cortina de seda",
# "el tiempo detenido") y el usuario la rechazó por lírica. Lo que quiere es que
# OPINE: un viejo verde que mira, compara con "antes las mujeres...", se relame,
# alecciona. La opinión es el tono; el CUERPO sigue saliendo de los hechos.
#
# DOS MEDICIONES SOBRE ESTE SYSTEM, no volver atrás sin leerlas:
#
#   1) El DETALLE FINO. Con la primera versión el redactor ya no inventaba
#      desnudo, pero ADORNABA: sobre un hecho que decía "bra negro con las manos
#      cerca del pecho" escribió "el pulgar presiona directamente contra el
#      pezón" y "una mano se apoya sobre los glúteos". Inventar anatomía fina es
#      lo que termina en tags falsos, así que la regla de abajo lo cubre.
#
#   2) El ECO. Cuando la lista de prohibiciones fue larga y enumerada, el modelo
#      empezó a CONTESTARLA dentro del texto: acro_0050 terminó con "sin añadir
#      ni un solo detalle más allí donde no estaba presente en los hechos dados"
#      y acro_0247 con "no hay rastro de otros cuerpos ni objetos visibles más
#      allá de lo esencial". Cuanto más enumerada la regla, más la devuelve.
#
# Por eso el system es CORTO y dice el límite en general en vez de enumerarlo.
# Con la voz del tío la opinión está permitida y pedida: lo único que NO se
# permite es afirmar cuerpo, acto o anatomía que no estén en los hechos.
_RED_S = """Escribís el comentario de una foto con la voz de un tío borracho,
chapado a la antigua, mirando a una mujer que le gusta. No ves la foto: solo
tenés los HECHOS, y los hechos son TODO lo que sabés de ella.

Español rioplatense, presente, primera persona de viejo verde: opinás, comparás
("antes las mujeres...", "en mi época..."), te relamés con lo que te gusta,
aleccionás, te hacés el entendido. El cuerpo que hay lo mirás con hambre y sin
pudor, nombrás lo que ves con las palabras de la calle cuando los hechos lo
justifican.

Un párrafo de 5 a 7 frases, COMPLETO: terminá la última frase. El morbo está en
cómo mirás lo que hay, no en lo que le agregás: contás la escena que los hechos
describen, con el hambre de un viejo que ya vio todo y se sorprende igual."""


def _hechos(image_path, model=None):
    """Pasada 1: mira la foto y anota hechos cortos. Sin redactar."""
    return (vlm.describe(str(image_path), model=model or MIRA, prompt=_MIRA_P,
                         system=_MIRA_S, num_predict=200,
                         temperature=0.3) or "").strip()


def _ficha(hechos, anclas="", nombre="", model=None, system=None):
    """Pasada 2: redacta desde los hechos. No mira la foto (generar_texto no
    manda imagen: es lo que garantiza que no agregue lo que no está).

    El dato anclado se presenta como un hecho MÁS, sin imperativos ("esto
    manda", "respetá esto"): se midió que el modelo contesta el imperativo
    dentro de la ficha ("no hay rastro de otros cuerpos"). Un dato más en la
    lista, en cambio, lo escribe sin comentarlo — y como el anclado del cuerpo
    dice el estado real ("cubiertos por la ropa"), corrige al VLM sin prohibir
    nada.

    `system` permite cambiar el REGISTRO sin tocar la mecánica: el default es
    _RED_S (prosa de fototeca); se probó con variantes de cuento erótico. Cambia
    la voz, no los hechos."""
    pr = "HECHOS de la foto:\n\n%s\n\n" % hechos
    if anclas:
        pr += "Datos de los clasificadores:\n%s\n\n" % anclas
    if nombre:
        pr += "La persona es %s.\n\n" % nombre
    pr += "Redactá la ficha ahora, en un párrafo completo."
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
