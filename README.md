# photo-sorter

Organizador y gestor de biblioteca de fotos con reconocimiento facial, **100% local**: agrupa los rostros de una carpeta, nombrás a cada persona, organizás a la **Biblioteca** (una carpeta por persona), y a partir de ahí tenés galerías, limpieza (duplicados, NSFW, caras separadas) y **producción de contenido** (pie de foto, comentarios, etiquetas, título/copete) generada por **modelos de IA que corren en tu máquina** (Ollama). Ninguna foto ni texto sale de la PC.

## De un vistazo (pestañas)

| Pestaña | Rutas | Qué es |
| --- | --- | --- |
| **Inicio** | `/inicio` | Tapa editorial de tu biblioteca (ver abajo). |
| **Explorar → Galería** | `/known` | Grid de personas conocidas → `/person/<nombre>` abre el visor de persona. |
| **Explorar → Rostros** | `/` | Grid de caras sueltas (todas las fotos de la biblioteca, agrupadas). |
| **Explorar → Persona** | `/finder` | Buscador de personas/parecidas. |
| **Limpieza → Duplicados** | `/dupes` | Encuentra fotos repetidas (SHA-1 exacto y dHash perceptual). |
| **Limpieza → Carpeta** | `/rename` | Renombra en masa (nomenclador). |
| **Limpieza → NSFW** | `/nsfw` | Escaneo con NudeNet. |
| **Limpieza → Separar caras** | `/split` | Parte una foto con mucha gente en recortes por persona. |
| **Contenido → Pies** | `/moldes` | Editor de moldes de pie de foto (sin IA, al instante). |
| **Contenido → Frases** | `/phrases` | Frases/descubrimientos para la portada. |
| **Contenido → Probador** | `/probador` | Pruébale etiquetas a una sola foto, en vivo. |
| **Contenido → Conversiones** | `/conversiones` | Diccionario de reemplazos de palabras (se usa en TODA la app). |
| **Sistema → Ajustes** | `/settings` | Modelos, textos de la IA, biblioteca, todo. |

Del menú original quedan: **Buscar duplicados**, **Renombrar**, **Formatos** e instalación en [SETUP.md](SETUP.md).

## Flujo general

```
PhotoSorter.bat  →  se abre http://127.0.0.1:5000  (también abre el navegador solo)
```

1. **Ajustes** → elegís la **Biblioteca** (carpeta que ya tiene una subcarpeta por persona; la home, la galería y el visor leen de acá).
2. **Analizar** desde **Rostros**: elegís una carpeta de fotos, detecta rostros (YuNet+SFace) y agrupa caras iguales (clustering, sin entrenamiento).
3. **Etiquetar**: cada grupo aparece con "¿quién es esta persona?"; los grupos sin nombre se ignoran.
4. **Organizar**: mueve cada foto a la **Biblioteca** (`<biblioteca>\<nombre>\`). Una foto con varias personas queda en la carpeta de cada una.

### Reconocimiento automático (personas conocidas)

- **Importar** desde una carpeta ya ordenada (una subcarpeta por persona) en el análisis.
- **Aprende sobre la marcha**: cada grupo que nombrás se suma a `data/known_people.json`.
- **Sugerencias** (💡): al analizar, los grupos parecidos a alguien conocido ya aparecen con el nombre escrito; confirmás con Enter o lo corregís.
- Umbral en `app/known.py` (`MATCH_EPS`, por defecto 0.50): más bajo = más estricto.

### Caras dudosas / depurar

Dentro de cada grupo, las caras que se despegan del centro se marcan con **⚠** (adaptativo, sin umbral fijo). Depurar: **✕ roja** por cara, o el botón **"🔍 Revisar y depurar"** en grupos grandes; las sacadas van a un grupo "Para revisar" — nunca se borra nada del disco.

## Duplicados, renombrar, NSFW, separar caras

- **Duplicados** (`/dupes`): exactas (SHA-1) y parecidas (dHash). Marca cuál conservar (mayor resolución/peso); las sobrantes se **mueven** a `_duplicados\` dentro de la carpeta.
- **Renombrar** (`/rename`): patrón `palabra_separador_0001.jpg`, con vista previa y cambio en dos pasos (no pisa nada).
- **NSFW** (`/nsfw`): escaneo NudeNet del contenido explícito.
- **Separar caras** (`/split`): corta cada persona de una foto grupal en su propio archivo.
- Más por consola: `python -m app.duplicates`.

## IA local (Ollama)

Todos los generadores (pie, comentarios, etiquetas, **visión** y **copete/título**) usan modelos que corren en **Ollama** (`127.0.0.1:11434`):

| Rol | Modelo típico | Dónde |
| --- | --- | --- |
| Etiquetas + Visión + Copete (mira la foto) | `Qwen2.5-VL-7B-NSFW-Caption-V3-abliterated (Q4_K_M)` (vlm.py `TAG_MODELS` / `COPETE_MODEL`) | Visor → selector de modelo; Ajustes |
| Pie y comentarios (texto) | `leeplenty/lumimaid-v0.2:12b` (`pie_model`/`comentarios_model`) | Ajustes |
| Texto por defecto | `huihui_ai/qwen3.5-abliterated:9b` (`post_model`, default `:4b` en vlm.py) | Ajustes |

Detalles en [SETUP.md](SETUP.md#3-ia-local-ollama-modelos-de-visión-y-texto): `KEEP_ALIVE=-1` (modelo residente en VRAM), `OLLAMA_MAX_LOADED_MODELS=1`. (Existe una **cola de trabajos en disco** pensada para procesar una tarea por vez sobre todas las fotos —`app/cola.py`, `data/cola.json`— que aguanta un corte de luz y retoma; está escrita pero todavía **sin enganchar** a la web: ver "features a medio hacer" más abajo.)

El **visor de persona** (`/person/<nombre>`) es la pieza central de contenido: cada foto tiene su pie, comentarios, etiquetas y favoritos, con un **Centro de IA** (🔮) que escribe la visión de la foto, las etiquetas sugeridas, el pie y los comentarios con el modelo elegido. En la pestaña **Pies** armás **moldes** (sin IA, instantáneos, en rioplatense y primera persona) **como herramienta aparte**: ya **no se disparan solos** al pedir el pie (antes interceptaban la generación y el botón "escribir con IA" devolvía un prehecho sin forma de salir de ahí). Ahora el pie lo escribe siempre el modelo.

Cada persona tiene **Nombre y alias** (modal ⚙ de su perfil, junto a la bio): un **nombre "lindo"** para mostrar (el título del álbum en la web sale de ahí; la carpeta y el reconocimiento facial no cambian) y sus **alias** como otras formas de llamarla. La IA usa el nombre casi siempre y de vez en cuando un alias, para no repetir el mismo nombre foto tras foto (en la Visión, el pie y el copete).


La **Visión** (paso 1) se guarda en dos partes: la **narración** (el texto redactado) y los **hechos** (lo que el VLM vio, técnico, antes de redactar), que se ven en un desplegable "👁 Lo que ve la IA" dentro del propio paso. El **sujeto del acto** —decir que ELLA es quien hace el acto— se afirma cuando su cara se reconoció por reconocimiento facial, o cuando **marcaste su cara a mano** en el visor, o cuando no hay ninguna otra persona identificada en la foto; si marcaste a otra, no se le atribuye.

Para marcar una cara a mano: en el visor, botón **🏷️ Caras** y **arrastrá** un cuadro sobre la cara (mientras arrastrás se ve el recuadro punteado amarillo; al soltar sale la lista de personas con avatar para elegir el nombre). El cuadro queda guardado en posición normalizada (sigue el zoom y el arrastre) y se borra con la **×** del nombre.


El botón **"Escribir los posts de todas sus fotos"** corre un lote de fondo (podés cerrar la página) que saltea lo ya hecho y retoma. Podés elegir qué generar: pie, comentarios, **visión** (lenta: ~2,5 min por foto), **copete**, o cualquier combinación.

El **pie** es una publicación de red social erótica **de venta de contenido**: vendés, provocás, te jactás de lo que mostrás y lo asumís ("te gusta lo que ves?", "soy tu putita"). No resume la foto (el que mira ya la ve), sino que suma lo que la foto no dice —ánimo, situación, clima, motivo— y **siempre le habla a él** (el que mira), arrancando por el vínculo ("hermano, mirá lo que me puse hoy"). El vínculo sale del perfil y se da vuelta con la tabla recíproca. En **Ajustes** podés cargar **tu nombre** (el del que mira) y **otros nombres con que ella te llama** (apodos): se usan como variante, el vínculo manda y el nombre o un apodo entran de vez en cuando. El **copete** es un título de tapa de revista, con el nombre y el vínculo adentro, y en clave erótica cuando la foto da ("10 formas de mantenerse con Paula", "Cómo tomar birra con Acro"). Los dos respetan la compuerta de edad: si la persona es `sano`, nada sexual.

## Portada premium (Inicio)

La pestaña **"Inicio"** (ruta `/inicio`) es una tapa editorial armada **solo con datos reales** de la app: nada de fechas, lugares ni colecciones inventados.

- **Hero carrusel**: empieza por las favoritas (`data/posts.json`); si no hay, usa fotos del archivo.
- **Protagonistas**: personas con más fotos, con su cuenta real.
- **Agregadas recientemente**: por fecha real de modificación (`mtime`), con etiquetas "Hoy"/"Ayer".
- **En este día**, **Colecciones** (por mes real + tarjeta "Favoritas") y **Descubrimientos** (curiosidades calculadas de los datos).
- Carga con **skeletons de carga** (shimmer) y el hero en `eager`; es de solo lectura.

Código: `app/home_premium.py` (`HOME_TEMPLATE` + `build_context()`); ruta: `app/flask_app.py` (`inicio()`).

## Mapa del proyecto

- **`app/flask_app.py`** (~450 KB): toda la web sobre Flask — rutas, plantillas, CSS y JS en un solo archivo. Nada se sirve con caché vieja: CSS/JS llevan hash `?v=` (para ver cambios: cerrá lanzador y **Ctrl+F5**).
- **Visión + texto (Ollama)**: `app/vlm.py` (VLM de etiquetado/visión, sistema de prompts), `app/posts.py` (pie/comentarios/copete + favoritos), `app/moldes.py` (editor de pies), `app/phrases.py`, `app/vision2.py` (paso 1 del Centro de IA: hechos + narración), `app/copetes.py` (titulares de noticia).
- **Features a medio hacer (listas para enganchar, sin rutas ni botones todavía)**: son módulos COMPLETOS cuyas tripas funcionan pero que nadie importa aún — no son código muerto, son features sin cara visible:
  - `app/narracion_ingles.py` (667 líneas): genera la narración/pie/comentarios **en inglés** (donde el vocabulario explícito es más denso) y los reescribe al rioplatense con el MISMO modelo + un endurecedor determinístico. Le faltan los tres botones del Creador ("Crear narración", "Crear pie", "Crear comentarios") y sus rutas en `flask_app.py`. Depende de `posts.narracion_de` y `vlm._crudo_map`, que ya existen (antes llamaba a funciones inexistentes y habría explotado al engancharse).
  - `app/cola.py` (210 líneas): cola de trabajos pesados persistente en disco (`data/cola.json`) que corre de a UNA tarea sobre TODAS las fotos, por fases, para cargar cada modelo de Ollama una sola vez y no mil veces. Retoma tras reinicio (`rescatar_huerfanos`), calcula ETA real. Le falta el worker (el hilo que saca de la cola) y las rutas `/cola/...` + el botón en la UI.
  - `app/similar.py`: búsqueda por parecido visual con CLIP (vectores de 512 números, distancia coseno) — distinto de Duplicados, que compara píxeles. Sin enganchar.
  - `app/texto.py`: separador de memes/capturas de texto por densidad de bordes con OpenCV puro (no hace OCR). Sin enganchar.
- **Análisis / biblioteca**: `app/analyze.py` (clustering), `app/facedet.py` (YuNet+SFace, modelos en `app/models/`), `app/known.py` (personas conocidas), `app/organizer.py`, `app/limpieza.py`, `app/corrector.py`, `app/renamer.py`, `app/splitter.py`, `app/notfaces.py`, `app/procesadas.py`, `app/respaldo.py`.
- **Escáneres**: `app/duplicates.py` (SHA-1/dHash), `app/nsfw.py` (NudeNet), `app/sexpos.py` (acto, resnet50), `app/finder.py` (buscador de personas).
- **Interface extra**: `app/home_premium.py` (portada), `app/tags.py`, `app/stats.py`, `app/person_meta.py`.
- **Datos**: `data/` (ver árbol en SETUP.md). Los `.json` se escriben foto a foto, nunca se pierde lo hecho.
- **Arranque**: `PhotoSorter.bat` (mata el puerto 5000 fantasma y lanza `venv\Scripts\python -m app.flask_app`; `PORT=5000`, `host=0.0.0.0`).

## Formatos de imagen

`.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp` y `.gif` (GIF por su primer fotograma). La extensión equivocada no importa: el formato se detecta por contenido. Detección: **YuNet** (detección) + **SFace** (reconocimiento) de OpenCV, rápidos en CPU (~1 s por foto), modelos en `app/models/`.

## Ajustes finos

En `app/analyze.py`, la constante `EPS` (por defecto `0.45`) controla qué tan parecidos deben ser dos rostros para caer en el mismo grupo:

- **Bajarla** (`0.40`) → más estricto: menos mezclas, pero una persona puede partirse en varios grupos (mismo nombre a todos → se unifican al organizar).
- **Subirla** (`0.50`) → más permisivo: riesgo de mezclar personas parecidas.

## Privacidad

- Reconocimiento, escaneos y generación corren 100% locales (OpenCV + Ollama).
- La web escucha en **`0.0.0.0:5000`** (accesible desde tu red local). Si no querés eso, cambiá el `host` en `app/flask_app.py` a `127.0.0.1`.
- Ninguna foto ni texto sale de tu máquina; los modelos de Ollama se bajan una única vez y después andan sin conexión.

Ver [SETUP.md](SETUP.md) para la instalación paso a paso y los respaldos.