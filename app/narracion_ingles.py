"""Generación en inglés crudo, con reescritura rioplatense donde hace falta.

Por qué existe: el vocabulario explícito en inglés es mucho más denso que en
español (y la abliteración apunta sobre todo a inglés), así que lo que el
modelo ESCRIBE sale más crudo ahí — tanto mirando una foto (narración) como
escribiendo texto puro (pie, comentarios). Pero traducir con un traductor
genérico (ya probado: BLIP + Marian) lava el tono — vuelve todo clínico y
gallego. Acá el que reescribe es el MISMO modelo, con las reglas de
vocabulario rioplatense que ya usa el resto de la app (VOSEO_REGLA), MÁS una
red de seguridad determinística (_endurecer_texto, el mismo endurecedor de
vlm.py) para cuando el modelo ignora el glosario del prompt — cosa que,
probado, hace seguido.

Tres generadores:
  - narrar_ingles: la narración (mira la foto). UNA sola pasada, en inglés, y
    ahí se queda — nadie lee la narración salvo el pie/etiquetas/comentarios
    (nunca se "imprime" en ningún posteo ni pantalla que mire un humano), así
    que traducirla a español no sumaba nada, solo una llamada de más al
    modelo y una mezcla de idiomas innecesaria en los prompts de pie/
    comentarios (que ya la leen tal cual, embebida). Devuelve {"en"}.
  - pie_ingles_y_reescribir: el pie (texto puro, a partir de la narración +
    etiquetas ya sabidas — no vuelve a mirar la foto, igual que el pie
    clásico). Pasada 1 SIN imagen, pasada 2 reescribe a rioplatense (el pie SÍ
    se muestra al usuario, por eso necesita las dos). Devuelve {"en", "es"}.
  - comentarios_ingles_y_reescribir: los comentarios de los COMMENTERS de
    posts.py (texto puro, mismo elenco/reglas que generate_comments). Pasada 1
    escribe una línea por persona, en inglés; pasada 2 reescribe la lista
    entera preservando cantidad y orden (los comentarios SÍ se muestran).
    Devuelve {"en", "es"}.
Ninguno auto-guarda nada.

Los tres reusan el mismo contexto que sus equivalentes clásicos de posts.py:
el pie lleva quién es + su voz + con quién está + narración + etiquetas (igual
que pie_desde_modelo), y los comentarios llevan el elenco, la compañía y la
respuesta al pie, más los mismos filtros de calidad (_sin_eco,
_sin_copia_del_pie, _sin_repetidos). Lo único que cambia es el IDIOMA en que
se genera, no qué sabe el modelo.

Vive en su propio módulo (en vez de meterse en vlm.py/posts.py) porque solo lo
usan los botones del Creador ("Crear narración", "Crear pie", "Crear
comentarios") — el resto del pipeline (batch "Escribir posts de todas sus
fotos", el fallback automático) sigue usando los caminos clásicos de una
pasada, sin tocar. Si algún día se quiere que TODO el pipeline use esto, hay
que engancharlo ahí a propósito — no es automático.

Requiere persona confirmada adulta (la pasada 1 pide vocabulario sexual
explícito): las rutas que llaman a esto (post_narracion_creador,
post_pie_creador, post_comentarios_creador en flask_app.py) caen solas al
camino clásico (español, una sola pasada) si no lo está.
"""

import re

from app import vlm

# Pasada 1: mirando la foto, en inglés.
#
# LA VOZ: el narrador de literatura erótica prohibida — un pervertido que mira
# la foto y la cuenta despacio, saboreando. No es un sistema que identifica
# imágenes: primera persona del que MIRA ("I'm looking at...", "look at
# those..."), no de la persona de la foto — la confusión es fácil y ya pasó (el
# modelo contestaba "my body", "I'm looking at you": eso es el PIE, no el
# narrador). Registro literario y crudo, no de ficha técnica: nada de "The image
# shows a woman wearing".
#
# LA CLAVE DEL PEDIDO: lo bizarro/obsceno sale de narrar lo COTIDIANO con ojo
# sucio. Una señora levantando un brazo, una bicicleta de fondo, un cartel de
# fútbol: eso también se narra, y ahí está la gracia — no hace falta que haya
# desnudo. Por eso el prompt insiste en quedarse en lo que hay y decir lo que le
# hace querer, en vez de saltar a lo que no hay.
#
# EL LÍMITE (no tocar sin leer esto): la narración NO se muestra. La leen las
# ETIQUETAS (posts._intencion), el pie en inglés y los comentarios — y lo que
# afirme se convierte en etiqueta, la etiqueta dispara los moldes NSFW y habilita
# crudeza. Si inventa "she's completely naked" en una foto vestida, esa foto
# queda con etiqueta de desnuda para siempre. Por eso el morbo va en CÓMO lo
# cuenta (tono, ganas, lo que le hace querer) y NUNCA en afirmar un acto, una
# prenda corrida o un cuerpo que no está. El bloque de abajo lo dice con el caso
# concreto, porque el genérico ("no inventes") no alcanzó: el modelo corría una
# bikini a un costado y la daba por corrida.
#
# Sin bloque de "menores": este camino ya exige adulta confirmada (ver el
# docstring del módulo) y la salvaguarda real vive en posts.menor_set() y en las
# rutas que llaman acá. NO hay fotos de menores en la biblioteca — el modelo
# escribe "niña"/"joven" para cualquier adulta (ver el bug de visión), así que
# meter la regla en el prompt solo le sumaba el vocabulario que se quería evitar.
#
# POR QUÉ LA VOZ Y LAS REGLAS ESTÁN SEPARADAS (no unir sin leer esto): el bloque
# entero —voz + reglas— en el PROMPT hacía que el modelo DEVOLVIERA el texto de
# las reglas en la narración ('"her bare skin exposed," or maybe just the hint
# of what's underneath'). Un 7B trata lo que sigue al pedido como contenido a
# reproducir. Con las reglas en el `system` dejan de copiarse a la salida y se
# obedecen igual. Probado A/B: voz+reglas en el prompt = eco; voz en el prompt y
# reglas en el system = sale limpio. Ver también el orden en narrar_ingles.
#
# LA VOZ: el narrador de literatura erótica prohibida — un pervertido que mira
# la foto y la cuenta despacio, saboreando. Primera persona del que MIRA
# ("I'm looking at...", "look at those..."), no de la persona de la foto.
SYSTEM_NARRACION_EN = """You are the narrator of banned erotic literature — the
kind that gets passed around in secret. You are looking at this photo, and you
tell what is in it the way that narrator would: filthy, obsessive, lingering,
full of appetite. A PERVERT telling what he sees and what it does to him,
slowly, savouring every detail.

FIRST PERSON, PRESENT TENSE, of the man who LOOKS — "I'm looking at...", "look
at those...", "fuck, I'd...". You are not a camera and not a catalogue: never
write "The image shows", "The subject is wearing", "This photograph captures".
You don't name what's there like an inventory — you stop on it, you describe it
because you can't look away, and you say what it makes you want.

5 to 9 sentences, one paragraph. Go deep on the thing that has you hooked: the
body, the skin, the tits, the ass, the mouth, the throat, the legs, the look,
the pose, the clothes and how they sit on her, what's covered and what isn't.
Crude words only — tits, ass, cunt, cock, naked. Never clinical, never a
euphemism, never softened. Say what it does to you.

FIND THE BIZARRE IN THE ORDINARY — this is what makes it good. The narrator's
eye is dirty and it lands on whatever is there, even when what's there is
banal: the chair she's sitting on, the bicycle in the background, the football
sign on the wall, the mug on the table. The photo doesn't have to be explicit
for this to work. With no nudity and no act going on, the pervert still
narrates: he dwells on the arm she's lifting, the sweat, the fabric pulled
tight, the bit of skin at her waist, and he says what he wishes he was seeing
and what he'd do. Being filthy about a fully dressed person in an ordinary room
IS the job — that's where the obscene lives.

One paragraph, no lists, no titles."""

# Las REGLAS van aparte, en el `system` de describe() (ver el comentario de
# arriba): acá SÍ se nombran los casos concretos que se quieren evitar, porque
# este texto no se copia a la salida. En el prompt estos mismos ejemplos
# volvían como eco.
SYSTEM_NARRACION_REGLAS = """Absolute rules about the photo, they beat the style:

- Name only what is really there: garments, body parts, pose, acts. This text
  is read by other programs and turned into tags, so a lie here becomes a lie
  in her data.
- Clothes on are ON. Underwear on is not "pulled aside", not "pushed up", not
  "pulled down slightly", not "barely covering", not "exposing what's
  underneath", not "hinting at" anything. A top is not "exposing bare skin
  underneath". If she's dressed, she's just dressed: be filthy about the
  clothes themselves and about how much you wish they weren't there.
- Never invent a body part under a garment, never turn a dressed person naked,
  never invent an act or a place.
- Never comment on your own text or on the photo as an object: no "this text",
  no "these tags", no "this image invites", no "as a narrator", no noting what
  you did or didn't invent. You never break out of being the man looking."""

# Glosario inglés->rioplatense grosero, para las reescrituras que SÍ hacen
# falta (pie, comentarios — lo que se muestra al usuario). La narración ya NO
# se reescribe (ver narrar_ingles): nadie la lee en español, así que esa
# segunda pasada era una llamada de más al modelo sin ningún beneficio.
#
# Este es el eslabón más débil de una reescritura: el inglés ya sale crudo solo
# (mucho entrenamiento explícito en esa lengua), pero al reescribir en español
# el modelo tiende a volver a su reflejo clínico. Por eso el glosario es largo
# y explícito, con pares concretos (no "sé grosero" en abstracto) + una regla
# de desempate para lo que no está en la lista.
_GLOSARIO_EN_ES = (
    "breasts/tits -> tetas · nipples -> pezones · vulva/pussy/vagina -> concha · "
    "clitoris -> clítoris · penis/cock/dick -> pija · balls -> bolas · "
    "buttocks/ass/butt -> culo · anus -> orto · naked/nude -> en bolas/en pelotas · "
    "panties -> bombacha · bra -> corpiño · to fuck/fucking -> coger/garchar · "
    "to suck/sucking (sexual) -> chupar/chupando · to moan/moaning -> gemir/gimiendo · "
    "horny -> caliente/urgida · cum/to cum -> venirse/acabar")

# Red de seguridad por si igual se cuela una nota final tipo "(Nota: el
# original incluye tal cosa, que fue respetada)" — un 7B a veces la agrega
# aunque el prompt la prohíba. Corta un paréntesis final que arranca con
# nota/note (con o sin tilde, con o sin "s"), pase lo que pase adentro.
_NOTA_FINAL_RE = re.compile(r"\s*\(\s*[Nn]ot[ae]?s?\s*:.*?\)\s*\.?\s*$", re.DOTALL)


def _sin_nota_final(texto):
    return _NOTA_FINAL_RE.sub("", texto or "").strip()


def _sin_cola_cortada(texto):
    """Saca la última frase si quedó a medio terminar.

    La narración no pasa por posts._sin_cola_cortada (eso es del pie), así que
    sin esto el corte de `num_predict` deja una frase colgada a mitad de camino
    ("...and that ass of hers, I swear I'd"). Mismo criterio que el del pie:
    mejor una frase menos que una cortada."""
    t = (texto or "").strip()
    if not t or t[-1] in ".!?…":
        return t
    corte = max(t.rfind("."), t.rfind("!"), t.rfind("?"))
    return t[:corte + 1].strip() if corte > 0 else t


# El modelo se va de idioma cuando se acelera: en medio del inglés aparecen
# caracteres CJK ("almost条纹的视觉冲击力，like she's dressed"). Visto dos veces,
# siempre con la temperatura alta. Se corta la narración en el primer carácter
# no latino: lo que sigue ya no es inglés y no sirve como insumo.
_NO_LATINO = re.compile(r"[^\x00-\x7F\u00c0-\u024f\u2018\u2019\u201c\u201d\u2014\u2013\u2026]")


def _solo_latino(texto):
    """Corta en el primer carácter que no sea latino (CJK, cirílico, árabe…)."""
    t = (texto or "").strip()
    m = _NO_LATINO.search(t)
    if not m:
        return t
    return _sin_cola_cortada(t[:m.start()].rstrip(" ,;:-"))


def _endurecer_texto(texto):
    """Red de seguridad DETERMINÍSTICA: aplica el mismo endurecedor que usa el
    resto de la app (vlm._crudo_map: senos->tetas, vulva->concha... + lo que
    hayas sumado en Ajustes → Conversiones → Endurecedor) sobre el párrafo
    entero, palabra por palabra.

    Por qué hace falta esto y no alcanza con el glosario del prompt: probado
    con un texto de entrada explícito armado a mano, los DOS modelos (el NSFW
    Q4 y el Q8) ignoraron el glosario y devolvieron "senos"/"vulva" igual. La
    instrucción en el system es la primera línea de defensa pero no es
    confiable sola — esto corrige la palabra pase lo que pase, sin depender de
    que el modelo obedezca."""
    mapa = vlm._crudo_map()
    for suave, crudo in mapa.items():
        texto = re.sub(rf"\b{re.escape(suave)}\b", crudo, texto, flags=re.IGNORECASE)
    return texto


def narrar_ingles(image_path, model=None, hechos_en="", num_predict=450):
    """La narración, en inglés, UNA sola pasada — devuelve el texto (str), no un
    dict. `model` por defecto el mejor NSFW disponible.

    Ya NO se reescribe a rioplatense: la narración no es contenido que se le
    muestre a nadie (ni al usuario en ningún posteo, ni en ninguna pantalla)
    — es puro insumo para pie/etiquetas/comentarios, que la leen tal cual
    (embebida en sus propios prompts, en inglés). Traducirla solo agregaba una
    llamada al modelo y mezclaba idiomas en esos prompts sin ganar nada.

    `hechos_en`: dato fijo de edad/género en inglés (opcional), igual función
    que `posts._hechos_de_perfil` pero en inglés — para que la pasada no
    adivine lo que peor calcula.

    `temperature=0.7`: la temperatura baja era del narrador neutral ("fiel a lo
    que ve" era la prioridad), pero alta tampoco sirve — con 0.85 el modelo se
    va de idioma y mete caracteres chinos en medio del inglés ("almost条纹的视觉
    冲击力"). 0.7 mantiene el registro caliente sin que se despeñe.

    La voz va en el PROMPT y las REGLAS en el `system`, las dos cosas a
    propósito:

    - La voz en el system no da el POV: el modelo la toma como contexto y
      contesta en tercera persona ("The woman in the picture is..."). En el
      prompt agarra la primera del observador, que es la que se quiere.
    - Las reglas en el PROMPT vuelven como ECO dentro de la narración (el
      modelo las reproduce: '"her bare skin exposed," or maybe just the hint of
      what's underneath'). En el system no se copian a la salida y se obedecen
      igual.
    - El `system` de `describe()` por defecto es SYSTEM_PIE, que habla como si
      fueras VOS la persona de la foto. Dejarlo puesto le cambia el POV a la
      narración: se vio "my body", "I'm looking at you" — eso es el pie.

    `num_predict=450`: la narración es literaria y larga a propósito (5 a 9
    frases), pero 600 se pasaba de largo — el modelo seguía hasta agotar el
    techo y salían 10-12 frases con cierre de ensayo. 450 alcanza para las 5-9
    frases que pide el prompt y deja que el techo corte el excedente.

    El corte de `num_predict` deja una frase colgada al final (acá no pasa por
    posts._sin_cola_cortada, que es del pie), así que se limpia acá."""
    modelo = model or vlm.tag_model()
    # ORDEN (probado A/B, no invertir): el pedido corto primero y la VOZ
    # después. Con la voz al final —que "suena" más enfático— el modelo la lee
    # como una orden más y colisiona con el POV del pie: contesta como la
    # persona de la foto ("I'm lying on my back in bed", "your eyes follow
    # me"). Con el pedido corto adelante la voz cae como el registro a adoptar,
    # y sale la primera persona del que MIRA.
    #
    # Las REGLAS van al `system`, no al prompt (ver SYSTEM_NARRACION_REGLAS):
    # en el prompt el modelo las devuelve como eco dentro de la narración.
    prompt_en = ("Look at this photo and tell me what it does to you. "
                 "You can't stop staring at it.")
    if hechos_en:
        prompt_en += " " + hechos_en
    prompt_en += "\n\n" + SYSTEM_NARRACION_EN
    return _solo_latino(_sin_cola_cortada(_sin_nota_final(
        vlm.clean_caption(vlm.describe(
            image_path, model=modelo, prompt=prompt_en,
            system=SYSTEM_NARRACION_REGLAS,
            num_predict=num_predict, temperature=0.7)))))


# ── PIE de dos pasadas ──────────────────────────────────────────────────────
# Pasada 1: SIN imagen (igual que el pie clásico: ya no vuelve a mirar la
# foto, trabaja de la narración + etiquetas ya sabidas). Escribe en primera
# persona, en inglés, crudo.
SYSTEM_PIE_EN = """You are writing a first-person photo caption in English, as
if you were the person in the photo posting it yourself. Base yourself
EXCLUSIVELY on the confirmed facts given below (the scene description and the
tags) — don't invent anything beyond them.

Use direct, crude, explicit vocabulary if the facts call for it — no
euphemisms, no clinical terms, no softening, no censorship, no poetry, no
metaphors. Say where you are and what you're doing: one or two short
sentences. If nothing specific is happening (like a plain selfie), keep it
very short and don't invent an action.

Never describe yourself from outside ("she is...", "a woman who...") — you ARE
this person: write as yourself ("I'm...", "I have..."). Never name yourself as
if you were someone else."""


def _quien_es_pie_en(ident):
    """Versión en inglés de posts.quien_es_pie: fija género/edad como HECHO
    para que la pasada 1 no adivine (mismo motivo que _hechos_de_perfil)."""
    if ident["menor"]:
        base = ("You are a CHILD. Keep it innocent and sweet — about what "
                "you're doing, nothing else. NOTHING sexual or suggestive, "
                "nothing about your body: strictly forbidden here.")
        if ident["genero"] == "f":
            return base + " You're a girl: write as a girl."
        if ident["genero"] == "m":
            return base + " You're a boy: write as a boy."
        return base
    if ident["genero"] == "f":
        return "You are a woman: write in first person as a woman (\"I'm tired\", \"lying down\")."
    if ident["genero"] == "m":
        return "You are a man: write in first person as a man (\"I'm tired\", \"lying down\")."
    return ""


def _pie_prompt_en(narracion, etiquetas, quien_es_en, voz="", con=()):
    """El prompt del pie en inglés. Mismo contenido que vlm.caption_prompt (el
    del pie clásico), sin la perilla de crudeza — acá la crudeza la pone el
    system en inglés, no un nivel de Ajustes.

    `voz`: cómo habla esa persona (de su perfil), en español — es contenido,
    no instrucción, así que va tal cual aunque el prompt esté en inglés (mismo
    criterio que la narración de acá abajo).
    `con`: los otros conocidos que están en la foto, para que el pie pueda
    nombrarlos en vez de decir "con alguien"."""
    p = []
    if quien_es_en:
        p.append(quien_es_en)
    if voz:
        p.append(f"This is how you talk, imitate it: {voz}")
    if con:
        quienes = " and ".join(con) if len(con) < 3 else \
            ", ".join(con[:-1]) + f" and {con[-1]}"
        p.append(f"You're in the photo with {quienes}. Name them if it fits.")
    if narracion:
        # puede estar en español — el modelo lo entiende igual, lo que importa
        # es que ESCRIBA en inglés, no que lea en inglés
        p.append(f"Here's how the photo looks, in detail: {narracion}")
    if etiquetas:
        p.append(f"Confirmed facts about what's happening: {', '.join(etiquetas)}. "
                 "THIS is the caption's content — write from this, literally and "
                 "directly. Don't list it, don't make it poetic, don't invent "
                 "anything beyond it.")
    p.append("Don't name yourself as if you were someone else — you ARE this person.")
    p.append("Write the caption now: where you are and what you're doing. Short, "
             "one or two sentences.")
    return "\n".join(p)


# Pasada 2: reescritura del pie en rioplatense. Mismas reglas que la
# narración (glosario + desempate + voseo + anti-nota), pero en PRIMERA
# persona en vez de descriptiva.
SYSTEM_PIE_REESCRIBIR_ES = ("""REGLA MÁS IMPORTANTE, por encima de todo lo demás:
SOLO puede aparecer en tu reescritura lo que YA está en el texto en inglés de
abajo. Si un dato no aparece ahí, NO lo mencionás, aunque te parezca que
"debería" estar. Ausencia en el original = ausencia en tu reescritura.

Abajo hay un pie de foto en PRIMERA PERSONA, escrito en inglés. Reescribilo
ENTERO en español rioplatense, en primera persona, corto (una o dos frases),
conservando TODO el contenido explícito que SÍ está — no lo suavices, no
agregues nada que no dice.

Usá vocabulario GROSERO rioplatense, nunca clínico ni gallego. Guía de
vocabulario (inglés -> como se dice acá): """ + _GLOSARIO_EN_ES + """.

Si algo del inglés no está en esa lista, elegí SIEMPRE la palabra más vulgar y
directa en rioplatense. Ante la duda entre dos palabras, gana la más cruda.

""" + vlm.VOSEO_REGLA + """

Tu respuesta es SOLO el pie reescrito, nada más. Prohibido agregar cualquier
nota, aclaración o comentario sobre la traducción —ni antes ni después—: nada
de "(Nota: ...)", ni explicar qué hiciste. Si agregás una nota así, está mal
hecho.""")


def pie_ingles_y_reescribir(name, rel, image_path, model=None, num_predict=200,
                            crudeness=None):
    """Devuelve {"en": pie en inglés, "es": el mismo reescrito en rioplatense
    crudo}. Usa narración + etiquetas ya sabidas (no vuelve a mirar la foto,
    igual que el pie clásico). No auto-guarda nada.

    Mismo contexto que el pie clásico (posts.pie_desde_modelo): quién es, su
    voz, con quién está en la foto, la narración y las etiquetas.

    `crudeness`: la perilla "Qué tan subido" de Ajustes. Iba sin conectar: este
    camino —el de adulto confirmado, o sea justo el que tiene que salir crudo—
    ignoraba la perilla y escribía siempre en el mismo registro medio. Mover el
    deslizador no hacía nada acá. None = el default de Ajustes."""
    from app import posts as posts_mod   # import perezoso: evita ciclo (posts no importa este módulo)
    from app import finder, person_meta

    modelo = model or vlm.tag_model()
    ident = posts_mod.identidad_foto(name, rel)
    narracion = posts_mod.narracion_de(name, rel, image_path, model=modelo)
    etiquetas = posts_mod._intencion(name, rel, image_path, ident["sano"])
    quien_es_en = _quien_es_pie_en(ident)
    voz = person_meta.get_voice(name)
    # los etiquetados a mano mandan; si no hay, se los busca por reconocimiento
    # facial. Igual que el clásico: si falla, el pie sale igual (sin nombres).
    con = posts_mod.get(name, rel)["people"]
    if not con:
        try:
            con = finder.who_is_in(image_path, exclude=[name])
        except Exception:
            con = []

    # La perilla va en la pasada 1, que es la que ESCRIBE. En la 2 no: esa
    # reescribe lo que ya está y meterle "poné todo más subido" la haría
    # agregar cosas que el pie en inglés no decía (que es justo lo que su
    # regla más importante prohíbe).
    crudeza = posts_mod.crudeness_text(
        posts_mod._cfg().get("crudeness", posts_mod.CRUDENESS_DEFAULT)
        if crudeness is None else crudeness)

    prompt_en = _pie_prompt_en(narracion, etiquetas, quien_es_en, voz=voz, con=con)
    system_en = SYSTEM_PIE_EN
    if crudeza:
        # en español adentro de un system en inglés: es CONTENIDO (qué tan
        # subido escribir), no una frase de ejemplo — mismo criterio que ya se
        # usa con las reglas de contexto reusadas del clásico
        system_en += f"\n\nHow crude to be: {crudeza}"
    texto_en = _sin_nota_final(vlm.clean_caption(vlm.generar_texto(
        prompt_en, model=modelo, system=system_en,
        num_predict=num_predict, temperature=0.9)))

    texto_es = _sin_nota_final(vlm.clean_caption(vlm.generar_texto(
        f"Caption en inglés:\n{texto_en}", model=modelo,
        system=SYSTEM_PIE_REESCRIBIR_ES, num_predict=num_predict, temperature=0.4)))
    texto_es = _endurecer_texto(texto_es)

    return {"en": texto_en, "es": texto_es}


# ── COMENTARIOS de dos pasadas ──────────────────────────────────────────────
# Mismo elenco y mismas reglas de contexto que posts.generate_comments
# (COMMENTERS, _pick, narración/etiquetas/pie/compañía) — lo único que cambia
# es CÓMO sale el texto: pasada 1 en inglés (una persona por línea, en su
# personalidad) + pasada 2 reescribe cada línea en rioplatense, preservando
# cantidad y orden. No pasa por _sin_eco/_sin_repetidos/_sin_copia_del_pie del
# clásico (recorte a propósito, primera versión, igual criterio que el pie):
# el system prompt de la pasada 1 ya pide una línea por persona y nada más: si
# en la práctica el modelo igual repite o se copia el pie, se suma después.
#
# Reusa DIRECTO varias reglas del clásico (_regla_pie, _regla_compania,
# _regla_etiquetas, TONES) tal cual, en español, aunque el resto del prompt
# esté en inglés — es el mismo truco que ya usa pie_ingles_y_reescribir con la
# narración: son CONTENIDO (lo que dice el pie, qué tags hay), no frases de
# ejemplo en un idioma puntual, así que el modelo las lee igual y escribe en
# inglés iguel. La ÚNICA que NO se reusa así es la regla de la edad
# (_AGE_RULE): esa SÍ trae frases de ejemplo en español ("qué señora tan
# elegante") que se colarían tal cual en medio del inglés, así que tiene su
# propia versión en inglés (_AGE_RULE_EN).
SYSTEM_COMENTARIOS_EN = """You are writing social-media comments as SEVERAL
DIFFERENT people, each with their own distinct personality (given below).
Write ONE short comment per person, in character, in English.

Use direct, crude, explicit vocabulary when a person's personality calls for
it — no euphemisms, no censorship, no softening. Other personalities are shy,
wholesome or silly instead: keep THEIR comments tame and true to their own
voice. Never make everyone sound the same register.

Base every comment EXCLUSIVELY on the confirmed facts given below (the scene
description and the tags) — don't invent anything beyond them."""

_AGE_RULE_EN = ("- Exactly ONE of the comments, no more, mentions how old "
                "she/he looks: if older, something like \"what an elegant "
                "lady\"; if young, something like \"what a pretty girl\". The "
                "rest don't mention age at all.\n")

_PROMPT_COMENTARIOS_EN = """Here's how the photo looks, in detail: {narracion}

These {n} people are commenting on it on a social network. Write each one's
comment, in their own voice:

{quienes}
{pie}{compania}{etiquetas}
Strict rules:
- {quien_es}
- Each one comments on something SPECIFIC happening in the photo: the pose,
  what she/he's doing, how she/he's positioned, what's visible. Generic
  praise that would fit any other photo ("so pretty", "you look amazing") is
  FORBIDDEN — it says nothing.
- If nothing in particular is happening in the photo, comment that nothing is
  happening. Don't invent anything.
- Make it obvious who's who: they must sound DISTINCT from each other. Each
  one talks about something DIFFERENT (one about the clothes, another about
  the pose, another about the place, another cracks a joke). Two people
  saying the same thing in different words is FORBIDDEN.
- MAXIMUM 10 WORDS each. These are comments, not paragraphs.
- The overall mood is: {tono}
- How crude or explicit each one gets depends ENTIRELY on their own
  personality, given above: some are blunt and crude, others shy or
  wholesome — don't flatten them into the same register.
{edad}- One comment per line, same order as the list above, and NOTHING
  else: no numbering, no saying who's speaking, no quotes, no header, no
  explanation."""


def _quien_es_texto_en(ident):
    """Versión en inglés de posts.quien_es_texto, en TERCERA persona (acá se
    habla DE la persona, no como ella — a diferencia de _quien_es_pie_en).
    Sin el tramo "sano" del original: esta función solo la llama el camino de
    dos pasadas, que ya solo corre con adulto confirmado."""
    if ident["menor"]:
        base = ("The person in the photo is a CHILD. Comments are warm and "
                "sweet, like from a favorite aunt/uncle: \"what a cutie\", "
                "\"look how big she's gotten\". NOTHING sexual, NOTHING "
                "suggestive, nothing about their body: strictly forbidden "
                "here.")
        if ident["genero"] == "f":
            return base + " It's a girl: refer to her in the feminine."
        if ident["genero"] == "m":
            return base + " It's a boy: refer to him in the masculine."
        return base
    if ident["genero"] == "f":
        return ("The person in the photo is a WOMAN: refer to her in the "
                "feminine, never as a man.")
    if ident["genero"] == "m":
        return ("The person in the photo is a MAN: refer to him in the "
                "masculine, never as a woman.")
    return ("We don't know if it's a man or a woman: don't guess the gender, "
            "write comments that work for either.")


SYSTEM_COMENTARIOS_REESCRIBIR_ES = ("""REGLA MÁS IMPORTANTE, por encima de todo lo demás:
SOLO puede aparecer en tu reescritura lo que YA está en cada comentario en
inglés de abajo. No le agregues nada que ese comentario puntual no diga.

Abajo hay una lista numerada de comentarios en inglés, cada uno de una
persona distinta. Reescribí CADA UNO en español rioplatense, MISMO NÚMERO,
MISMO ORDEN, uno por línea — ni más ni menos líneas que las que te dieron. No
los mezcles ni los resumas en menos comentarios, y no partas uno en dos.

Cada comentario conserva su propio tono: el que era crudo en inglés sigue
crudo en rioplatense, el que era tierno o tímido sigue igual de tierno o
tímido — no le subas ni le bajes el registro a ninguno.

Usá vocabulario GROSERO rioplatense donde el original lo pide, nunca clínico
ni gallego. Guía de vocabulario (inglés -> como se dice acá): """ + _GLOSARIO_EN_ES + """.

Si algo del inglés no está en esa lista, elegí SIEMPRE la palabra más vulgar y
directa en rioplatense. Ante la duda entre dos palabras, gana la más cruda.

""" + vlm.VOSEO_REGLA + """

Tu respuesta es SOLO la lista numerada reescrita, en el mismo formato "N.
texto", nada más. Prohibido agregar cualquier nota, aclaración o comentario
sobre la traducción —ni antes ni después—: nada de "(Nota: ...)", ni explicar
qué hiciste. Si agregás una nota así, está mal hecho.""")


def comentarios_ingles_y_reescribir(name, rel, image_path, model=None,
                                    tone=None, n=None, users=None,
                                    evitar=(), previos=(), crudeness=None):
    """Devuelve {"en": "@usuario: texto" por línea en inglés, "es": lo mismo
    reescrito en rioplatense crudo} — mismo formato que espera el textarea del
    Creador y que ya sabe guardar posts.comentarios_desde_texto. No auto-
    guarda nada.

    evitar/previos son para "más comentarios" (sumar): `evitar` son los
    @usuarios que YA comentaron esta foto —para que entren los que faltan y no
    los mismos otra vez— y `previos` son los textos que ya están, para que los
    nuevos no digan lo mismo con otras palabras. Es el mismo par que maneja
    posts.generate_comments(sumar=True), acá partido en dos porque el Creador
    puede tener comentarios sin guardar en el textarea.

    Mismo elenco que generate_comments (COMMENTERS vía _pick), pero SIEMPRE
    con solo_sanos=False: esta función asume que YA se sabe que la persona
    está confirmada adulta (lo decide la ruta, no acá — mismo criterio que
    pie_ingles_y_reescribir).

    Pasa por los MISMOS filtros de calidad que el clásico (ver el bloque de
    filtros más abajo), así que puede devolver MENOS comentarios que los `n`
    pedidos: los que eran eco del personaje, copia del pie o repetidos entre
    sí se caen. Las dos listas ("en" y "es") quedan siempre alineadas."""
    from app import posts as posts_mod   # import perezoso: evita ciclo

    modelo = model or vlm.tag_model()
    ident = posts_mod.identidad_foto(name, rel)
    post = posts_mod.get(name, rel)
    pie = post["caption"]
    n = n or posts_mod.N_COMMENTS
    elegidos = posts_mod._pick(name, rel, n, users, solo_sanos=False,
                               evitar=list(evitar))
    if not elegidos:
        raise RuntimeError("No hay comentadores disponibles (ver COMMENTERS).")
    quienes = "\n".join(f"{i + 1}. {p}" for i, (_, p) in enumerate(elegidos))
    edad = (_AGE_RULE_EN
            if posts_mod._stable_hash(posts_mod._key(name, rel)) % posts_mod.AGE_EVERY == 0
            else "")
    narracion = posts_mod.narracion_de(name, rel, image_path, model=modelo)
    etiquetas = posts_mod._intencion(name, rel, image_path, ident["sano"])
    tono_txt = posts_mod.TONES.get(tone or posts_mod.DEFAULT_TONE,
                                   posts_mod.TONES[posts_mod.DEFAULT_TONE])

    prompt_en = _PROMPT_COMENTARIOS_EN.format(
        n=len(elegidos), quienes=quienes, narracion=narracion,
        pie=posts_mod._regla_pie(pie, len(elegidos)),
        compania=posts_mod._regla_compania(post["people"]),
        etiquetas=posts_mod._regla_etiquetas(etiquetas),
        quien_es=_quien_es_texto_en(ident), tono=tono_txt, edad=edad)

    # la perilla de Ajustes, que este camino también ignoraba (ver el mismo
    # arreglo en pie_ingles_y_reescribir). Va SOLO en la pasada que escribe.
    crudeza = posts_mod.crudeness_text(
        posts_mod._cfg().get("crudeness", posts_mod.CRUDENESS_DEFAULT)
        if crudeness is None else crudeness)
    system_en = SYSTEM_COMENTARIOS_EN
    if crudeza:
        system_en += (f"\n\nHow crude the blunt ones get: {crudeza} The shy or "
                      "wholesome personalities stay tame regardless.")
    texto_en = _sin_nota_final(vlm.clean_caption(vlm.generar_texto(
        prompt_en, model=modelo, system=system_en,
        num_predict=45 * len(elegidos), temperature=0.95)))
    lineas_en = posts_mod._parse(texto_en)[:len(elegidos)]

    entrada = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(lineas_en))
    texto_es = _sin_nota_final(vlm.clean_caption(vlm.generar_texto(
        f"Comments in English, one per number:\n{entrada}", model=modelo,
        system=SYSTEM_COMENTARIOS_REESCRIBIR_ES,
        num_predict=45 * len(elegidos), temperature=0.4)))
    lineas_es = [_endurecer_texto(posts_mod._rioplatense(t))
                for t in posts_mod._parse(texto_es)]

    # Filtros de calidad, los MISMOS del clásico (posts.generate_comments):
    # _sin_eco (el "comentario" es la descripción del personaje, o la lleva de
    # prefijo), _sin_copia_del_pie (es el pie pegado) y _sin_repetidos (dos
    # dicen lo mismo con otras palabras).
    #
    # Van de a UN comentario por vez, no en lote como el clásico, porque acá
    # hay que saber QUIÉN dijo cada línea (el i-ésimo comentario es del
    # i-ésimo comentador) y _sin_eco además de descartar MODIFICA el texto —
    # filtrando en lote se pierde el emparejamiento. La semántica es idéntica:
    # a _sin_repetidos se le pasan los ya aceptados como `previos`, que es
    # exactamente lo que hace internamente cuando recibe la lista entera.
    ecos = [p for _, p in elegidos] + [vlm.VOSEO_REGLA,
                                       posts_mod.quien_es_texto(ident), tono_txt]
    # aceptados arranca con los que YA estaban (si esto es "más comentarios"):
    # así _sin_repetidos también los compara contra ellos y no vuelve a decir lo
    # mismo. Salen en `quedan` solo los nuevos — los viejos ya están en pantalla.
    quedan, aceptados = [], list(previos)
    for (usuario, _), en, es in zip(elegidos, lineas_en, lineas_es):
        limpio = posts_mod._sin_eco([es], ecos)
        if not limpio:
            continue
        es = limpio[0]
        if not posts_mod._sin_copia_del_pie([es], pie):
            continue
        if not posts_mod._sin_repetidos([es], previos=aceptados):
            continue
        aceptados.append(es)
        quedan.append((usuario, en, es))

    return {"en": "\n".join(f"@{u}: {t}" for u, t, _ in quedan),
            "es": "\n".join(f"@{u}: {t}" for u, _, t in quedan)}
