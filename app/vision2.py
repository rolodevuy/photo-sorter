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
  2) REDACTAR (30B, NO mira la foto): escribe la ficha desde los hechos. Un 30B
     respeta un límite que un 7B no: no agregó nada que no estuviera en la lista.

El dato anclado (acto, ropa cubierta, nombre) va en la pasada 2, que es la que lo
respeta. En la 1 se pierde.

COSTO: ~2,3 minutos por foto (el 30B tarda ~130s en CPU). Por eso esto NO corre
en tandas automáticas: es un botón, una foto, y queda guardado.
"""

from app import anclados, vlm

# El que MIRA. Es el VLM que ya usa la app para el pie y las etiquetas.
MIRA = "hf.co/mradermacher/Qwen2.5-VL-7B-NSFW-Caption-V3-abliterated-GGUF:Q4_K_M"

# El que REDACTA. Solo texto, sin foto: por eso no puede inventar mirando.
REDACTOR = "huihui_ai/qwen3-abliterated:30b-a3b-instruct-2507-q4_K_M"

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
# DOS MEDICIONES SOBRE ESTE SYSTEM, no volver atrás sin leerlas:
#
#   1) El DETALLE FINO. Con la primera versión el 30B ya no inventaba desnudo,
#      pero ADORNABA: sobre un hecho que decía "bra negro con las manos cerca
#      del pecho" escribió "el pulgar presiona directamente contra el pezón" y
#      "una mano se apoya sobre los glúteos". Se agregó la prohibición nombrada
#      del detalle (contacto, textura, tamaño) y eso bajó el adorno.
#
#   2) El ECO. Pero con la lista larga de prohibiciones el modelo empezó a
#      CONTESTARLAS dentro de la ficha: acro_0050 terminó con "sin añadir ni un
#      solo detalle más allí donde no estaba presente en los hechos dados" y
#      acro_0247 con "no hay rastro de otros cuerpos ni objetos visibles más allá
#      de lo esencial". El texto final citaba la instrucción — el mismo eco que
#      ya se había visto con el narrador. Cuanto más larga y enumerada la regla,
#      más la devuelve.
#
# Por eso este system dice lo mismo que el de la medición 1, pero CORTO y sin
# enumerar: una regla general de "los hechos son todo" en vez de una lista de
# cosas prohibidas que el modelo pueda repetir. Lo enumerado vivía en el prompt
# y se copiaba; acá se probó que también se copia desde el system.
_RED_S = """Sos el ojo de una fototeca y redactás la ficha de una foto a partir
de los HECHOS que te dan. No ves la foto: solo tenés los hechos.

Los hechos son TODO lo que sabés. Un párrafo de 4 a 6 frases, en español
rioplatense, tercera persona, presente, COMPLETO: terminá la última frase.

Contás cada hecho una vez, con ritmo y detalle, sin repetirlo ni opinar.
Escribís solo lo que los hechos dicen, con las palabras que usan."""


def _hechos(image_path, model=None):
    """Pasada 1: mira la foto y anota hechos cortos. Sin redactar."""
    return (vlm.describe(str(image_path), model=model or MIRA, prompt=_MIRA_P,
                         system=_MIRA_S, num_predict=200,
                         temperature=0.3) or "").strip()


def _ficha(hechos, anclas="", nombre="", model=None):
    """Pasada 2: redacta desde los hechos. No mira la foto (generar_texto no
    manda imagen: es lo que garantiza que no agregue lo que no está).

    El dato anclado se presenta como un hecho MÁS, sin imperativos ("esto
    manda", "respetá esto"): se midió que el modelo contesta el imperativo
    dentro de la ficha ("no hay rastro de otros cuerpos"). Un dato más en la
    lista, en cambio, lo escribe sin comentarlo — y como el anclado del cuerpo
    dice el estado real ("cubiertos por la ropa"), corrige al VLM sin prohibir
    nada."""
    pr = "HECHOS de la foto:\n\n%s\n\n" % hechos
    if anclas:
        pr += "Datos de los clasificadores:\n%s\n\n" % anclas
    if nombre:
        pr += "La persona es %s.\n\n" % nombre
    pr += "Redactá la ficha ahora, en un párrafo completo."
    return (vlm.generar_texto(pr, model=model or REDACTOR, system=_RED_S,
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
