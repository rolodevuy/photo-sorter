"""Sugeridor de etiquetas por imagen (100% local, offline tras la 1ra descarga).

Dos cosas de una pasada sobre la foto:
- FRASE: una descripción en lenguaje natural (BLIP large).
- TAGS: de TU vocabulario (etiquetas ya usadas + atajos), CLIP puntúa cuáles
  aplican mejor a la imagen (zero-shot). No inventa palabras nuevas: rankea las
  tuyas.

Los modelos (BLIP + CLIP) se bajan de Hugging Face UNA vez (varios GB) y después
todo corre local, sin conexión. Se cargan perezosamente (recién en el primer
uso) porque ocupan bastante RAM y tardan en levantar.

Pesado y lento en CPU (segundos por foto): pensado para usarse de a una foto,
a pedido, en el etiquetador — no en masa.
"""

import threading

import numpy as np
from PIL import Image, ImageOps

BLIP_MODEL = "Salesforce/blip-image-captioning-large"
CLIP_MODEL = "openai/clip-vit-base-patch32"
TRANS_MODEL = "Helsinki-NLP/opus-mt-en-es"   # traductor inglés->español (local, chico)
VQA_MODEL = "Salesforce/blip-vqa-base"       # responde preguntas sobre la imagen

# Preguntas sobre la persona (el modelo solo entiende inglés; la etiqueta es lo
# que ves vos). Se responden de a una: ~3s cada una.
PERSON_QUESTIONS = [
    ("pelo", "what color is the hair of the person?"),
    ("ropa", "what is the person wearing?"),
    ("pose", "what is the person doing?"),
    ("expresión", "what is the facial expression of the person?"),
]

_lock = threading.Lock()   # cargar/usar los modelos: uno a la vez (no thread-safe)
_blip = None
_blip_proc = None
_clip = None
_clip_proc = None
_trans = None
_trans_tok = None
_vqa = None
_vqa_proc = None


def _load():
    global _blip, _blip_proc, _clip, _clip_proc
    if _blip is None:
        import torch
        from transformers import (BlipForConditionalGeneration, BlipProcessor,
                                   CLIPModel, CLIPProcessor)
        _blip_proc = BlipProcessor.from_pretrained(BLIP_MODEL)
        _blip = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL)
        _blip.eval()
        _clip_proc = CLIPProcessor.from_pretrained(CLIP_MODEL)
        _clip = CLIPModel.from_pretrained(CLIP_MODEL)
        _clip.eval()
    return _blip, _blip_proc, _clip, _clip_proc


def _load_vqa():
    global _vqa, _vqa_proc
    if _vqa is None:
        from transformers import BlipForQuestionAnswering, BlipProcessor
        _vqa_proc = BlipProcessor.from_pretrained(VQA_MODEL)
        _vqa = BlipForQuestionAnswering.from_pretrained(VQA_MODEL)
        _vqa.eval()
    return _vqa, _vqa_proc


def ask(image, question):
    """Le pregunta algo a la imagen (BLIP-VQA). Devuelve la respuesta en inglés."""
    import torch
    model, proc = _load_vqa()
    inputs = proc(image, question, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=20)
    return proc.decode(out[0], skip_special_tokens=True).strip()


def person_details(image, questions=None):
    """Detalles de la persona: [(etiqueta, respuesta en español), ...].

    Pregunta de a una (pelo, ropa, pose, expresión). Descarta respuestas vacías
    o inútiles ("unanswerable", "no"). ~3s por pregunta."""
    out = []
    for label, q in (questions or PERSON_QUESTIONS):
        try:
            a = ask(image, q)
        except Exception:
            continue
        if not a or a.lower() in ("unanswerable", "unknown", "no", "none", "nothing"):
            continue
        out.append((label, translate_es(a)))
    return out


def _load_translator():
    global _trans, _trans_tok
    if _trans is None:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        _trans_tok = AutoTokenizer.from_pretrained(TRANS_MODEL)
        _trans = AutoModelForSeq2SeqLM.from_pretrained(TRANS_MODEL)
        _trans.eval()
    return _trans, _trans_tok


def translate_es(text):
    """Traduce del inglés al español (local). Si algo falla, devuelve el original."""
    import torch
    try:
        model, tok = _load_translator()
        inputs = tok([text], return_tensors="pt", truncation=True)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=60)
        return tok.decode(out[0], skip_special_tokens=True).strip()
    except Exception:
        return text


def is_ready():
    """True si los modelos ya están cargados en memoria (no dispara la carga)."""
    return _blip is not None


def _open(path):
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def caption(image):
    """Frase descriptiva de la imagen (BLIP), sin repetirse en loop."""
    import torch
    blip, proc, _, _ = _load()
    inputs = proc(image, return_tensors="pt")
    with torch.no_grad():
        out = blip.generate(**inputs, max_new_tokens=40, min_new_tokens=5,
                            num_beams=3, no_repeat_ngram_size=2,
                            repetition_penalty=1.4)
    return proc.decode(out[0], skip_special_tokens=True).strip()


def score_tags(image, vocab, top_n=12, min_prob=0.55):
    """Puntúa qué etiquetas de `vocab` aplican a la imagen (CLIP zero-shot,
    MULTI-etiqueta: cada tag por separado, no 'elegí uno').

    Para cada tag compara "a photo of <tag>" contra un genérico "a photo": la
    probabilidad es cuánto MÁS describe la foto el tag que el genérico. Devuelve
    [(tag, prob 0-1), ...] con los que superan min_prob, ordenados; si ninguno,
    los mejores 3 igual (para que siempre sugiera algo que revisás)."""
    import torch
    if not vocab:
        return []
    _, _, clip, proc = _load()
    texts = [f"a photo of {t}" for t in vocab] + ["a photo"]
    inputs = proc(text=texts, images=image, return_tensors="pt",
                  padding=True, truncation=True)
    with torch.no_grad():
        logits = clip(**inputs).logits_per_image[0]   # 1 por texto (con temperatura)
    generic = logits[-1]
    # softmax binaria tag-vs-genérico -> prob independiente por tag (multi-label)
    probs = torch.softmax(torch.stack([logits[:-1], generic.expand(len(vocab))], dim=1), dim=1)[:, 0]
    probs = probs.cpu().numpy()
    ranked = sorted(zip(vocab, probs), key=lambda x: x[1], reverse=True)
    keep = [(t, float(p)) for t, p in ranked if p >= min_prob]
    return keep[:top_n] if keep else [(t, float(p)) for t, p in ranked[:3]]


def interrogate(image_path, vocab, with_details=True, engine="blip",
                vlm_model=None, person=None):
    """Frase + tags + detalles de la persona, en una pasada. Serializado (los
    modelos no son thread-safe).

    engine="blip": frase con BLIP + detalles con VQA. Rápido, pero le pifia, y
    la frase sale en inglés y hay que traducirla (queda neutra y planchada).
    engine="vlm": el pie de foto con un modelo de visión en Ollama. Lento, mucho
    mejor, y escribe DIRECTO en español, así que no pasa por el traductor: el
    tono crudo/uruguayo no sobrevive a una traducción automática.

    `person` es de quién es la foto (sale de la carpeta): el VLM la nombra en vez
    de adivinar.

    Los tags siempre los pone CLIP: son de TU vocabulario, no dependen del motor.

    Devuelve {"caption": frase en español, "caption_en": original en inglés (o
    None si ya vino en español), "tags": [(tag, score), ...],
    "details": [(etiqueta, respuesta), ...]}."""
    with _lock:
        img = _open(image_path)
        if engine == "vlm":
            from app import vlm
            cap_es = vlm.clean_caption(
                vlm.describe(image_path, model=vlm_model or vlm.DEFAULT_MODEL,
                             prompt=vlm.caption_prompt(person)))
            cap_en = None         # nunca hubo inglés: nació en español
            details = []          # el pie ya cuenta ropa, cuerpo y acciones
        else:
            cap_en = caption(img)
            cap_es = translate_es(cap_en)
            details = person_details(img) if with_details else []
        return {"caption": cap_es, "caption_en": cap_en,
                "tags": score_tags(img, vocab),
                "details": details}
