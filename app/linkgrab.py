"""Extractor de enlaces: carga una página web y trae los enlaces que
contengan el texto que elijas.

Usa un navegador real (el Edge de Windows, invisible, vía Playwright), así
también funciona en páginas que arman su contenido con JavaScript. Busca el
texto en el CÓDIGO de la página ya renderizada: atributos href/src y
cualquier URL suelta en el HTML (scripts, galerías, etc.). Cada página es
distinta, por eso el texto a buscar lo elegís vos en cada búsqueda.

Es la única parte del programa que se conecta a internet, y solo a la
dirección que vos le des.
"""

import html as html_mod
import re

from playwright.sync_api import sync_playwright

# URLs sueltas en el código (comillas, espacios o <> las terminan)
URL_RE = re.compile(r"https?://[^\s\"'<>\\]+")

# Scrolls al fondo para disparar la carga diferida (galerías que van
# agregando contenido a medida que bajás). Más scrolls = más resultados en
# páginas infinitas, pero más espera.
SCROLLS = 4
SCROLL_WAIT_MS = 700
PAGE_TIMEOUT_MS = 30000


def _clean(url):
    """Normaliza una URL encontrada en el código (entidades HTML, restos)."""
    return html_mod.unescape(url).strip().rstrip(".,;)")


def grab_links(page_url, term):
    """Devuelve la lista (sin repetidos, en orden de aparición) de URLs de la
    página que contienen `term` (sin distinguir mayúsculas).

    Lanza RuntimeError con un mensaje entendible si la página no se puede
    cargar o no está el navegador.
    """
    term_l = term.strip().lower()
    try:
        with sync_playwright() as p:
            try:
                browser = p.chromium.launch(channel="msedge", headless=True)
            except Exception as e:
                raise RuntimeError(
                    "No se pudo abrir el navegador Edge de Windows "
                    f"(hace falta para leer páginas con JavaScript): {e}")
            try:
                page = browser.new_page()
                page.goto(page_url, wait_until="domcontentloaded",
                          timeout=PAGE_TIMEOUT_MS)
                try:  # esperar a que termine de cargar, sin colgarse si no para
                    page.wait_for_load_state("networkidle", timeout=8000)
                except Exception:
                    pass
                for _ in range(SCROLLS):
                    page.evaluate("window.scrollBy(0, document.body.scrollHeight)")
                    page.wait_for_timeout(SCROLL_WAIT_MS)
                # href/src ya resueltos a URL absoluta por el navegador
                attrs = page.eval_on_selector_all(
                    "[href], [src]",
                    "els => els.map(e => e.href || e.src).filter(Boolean)")
                source = page.content()
            finally:
                browser.close()
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"No se pudo cargar la página: {e}")

    found, seen = [], set()
    for raw in list(attrs) + URL_RE.findall(source):
        url = _clean(raw)
        if term_l in url.lower() and url not in seen:
            seen.add(url)
            found.append(url)
    return found
