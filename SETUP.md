# Guía de instalación y puesta en marcha

Documentación para **replicar la app desde cero** en otra máquina (o acá
mismo, si algún día se vuelve a necesitar). Todo lo que hagas en la instalación
queda local: ninguna foto ni texto sale de la máquina.

---

## 0. Requisitos de la máquina

| Qué | Recomendado | Notas |
| --- | --- | --- |
| SO | Windows 10/11 | Los scripts de arranque (`.bat`) y el selector de carpetas son de Windows. |
| Python | 3.10+ | Probado con 3.11/3.12. |
| CPU / RAM | 8 GB+ | Para análisis facial, detección, duplicados. |
| **GPU** | NVIDIA con **12–16 GB de VRAM** (usamos una RTX 5060 Ti 16 GB) | Si no hay GPU, la app corre pero la generación con IA es 10–40× más lenta. |
| Ollama | >= 0.7 | Server local de modelos (ver sección 3). |

> Si no tenés GPU NVIDIA o no querés los modelos, la app **sigue andando**:
> solo las funciones de IA (pie, comentarios, etiquetas, visión, copete) quedan
> deshabilitadas o muy lentas.

---

## 1. Instalar la app (una sola vez)

### Opción A: `instalar.bat` (doble clic)
Crea el entorno, instala `requirements.txt` y lista los modelos ONNX de
detección. Necesita internet **solo esta vez**.

### Opción B: manual (equivalente)

```bash
cd photo-sorter
python -m venv venv
venv\Scripts\python -m pip install --upgrade pip
venv\Scripts\python -m pip install -r requirements.txt
```

**Nota sobre `torch`:** comes para el clasificador de acto (`sexpos.py`,
opcional). El `requirements.txt` trae `torch>=2.2` (wheels CUDA de Windows).
Si vas a correr **solo en CPU**, instalalo así para no bajar 2 GB de CUDA:

```bash
venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Los modelos de **detección de rostros** (`app/models/yunet.onnx` y
`sface.onnx`) vienen incluidos en el repo, no hay que descargar nada.

---

## 2. Correr la app

```bash
PhotoSorter.bat
```

O en su defecto:

```bash
venv\Scripts\python -m app.flask_app
```

- Abre la web en **http://127.0.0.1:5000** (el lanzador también abre el
  navegador solo).
- Puerto Flask: **5000**. Puerto de Ollama: **11434**.
- Para terminar: cerrar la ventana negra.
- **Importante:** `PhotoSorter.bat` mata cualquier proceso fantasma que quede
  escuchando en el puerto 5000 antes de arrancar. Si cambiás el código y "no
  ves los cambios", es porque sigue corriendo una instancia vieja: cerrá el
  lanzador y abrí de nuevo, o recargá el navegador con **Ctrl+F5**.

---

## 3. IA local: Ollama (modelos de visión y texto)

La generación de pie, comentarios, etiquetas, visión y copete usa modelos que
corren en **Ollama** (servidor local en `127.0.0.1:11434`). Instalá los que
quieras con, por ejemplo:

```powershell
ollama pull hf.co/mradermacher/Qwen2.5-VL-7B-NSFW-Caption-V3-abliterated-GGUF:Q4_K_M
ollama pull leeplenty/lumimaid-v0.2:12b
ollama pull huihui_ai/qwen3.5-abliterated:9b
```

Roles (dónde los usa la app):

| Rol | Modelo recomendado | Dónde se elige |
| --- | --- | --- |
| Etiquetas + Visión + Copete (mira la foto) | `hf.co/mradermacher/Qwen2.5-VL-7B-NSFW-Caption-V3-abliterated-GGUF:Q4_K_M` | Visor → selector de modelo, y Ajustes. |
| Pie y comentarios (texto) | `leeplenty/lumimaid-v0.2:12b` | Ajustes → `pie_model`/`comentarios_model`. |
| Pie/comentarios por defecto | `huihui_ai/qwen3.5-abliterated:9b` | Ajustes → `post_model`. |

Configuración clave (en `app/vlm.py`):

- `KEEP_ALIVE = "-1"`: el modelo queda **residente en VRAM** (si no, cada
  generación paga la carga del modelo). Con 16 GB de VRAM el 7B de visión vive
  siempre cargado.
- `TAG_MODELS`: orden de preferencia para etiquetar (baja al primer modelo de
  la lista que esté instalado).
- Modelos con 🔞 en el selector = sin censura (por nombre: `abliterated`,
  `nsfw`, etc.).

**Cola de trabajos (`app/cola.py`).** Los trabajos pesados (narrar, etiquetar,
pies, comentarios de TODAS las fotos) se encolan **uno a la vez y en disco**
(`data/cola.json`). Razón: Ollama con `OLLAMA_MAX_LOADED_MODELS=1` descarga el
modelo en uso cuando carga otro; procesando por-fase cada modelo se carga UNA
vez. Podés dejarla toda la noche: si se corta la luz retoma donde iba.

---

## 4. `data/config.json` (la memoria de la app)

Es el único archivo que conviene respaldar/editar a mano (los datos salen en
git ignore). Claves principales:

| Clave | Qué hace |
| --- | --- |
| `library` | Carpeta **Biblioteca**: un subdirectorio por persona (la galería y la home leen de acá). |
| `photos_dir` | Carpeta de origen que se analiza (se mueven fotos a la Biblioteca al organizar). |
| `min_face`, `media` | Tamaño mínimo de cara y qué revisar en el análisis. |
| `known_dir` / `output_dir` | Legacy de la versión original (Biblioteca). |
| `post_model`, `tag_model`, `narracion_model`, `pie_model`, `comentarios_model` | Modelos de Ollama por rol. |
| `post_extra_*`, `sys_pie`, `sys_tags`, `sys_comentarios`, `comment_users`, `crudeness`, `comment_tone` | Textos/reglas con que la IA escribe el pie, las etiquetas y los comentarios. |
| `f_conversiones` | Diccionario de reemplazos (pija rusa → paja rusa, etc.), se edita en la pestaña Conversiones. |
| `diminutivos`, `f_acto`, `genero_duenio`, `vista_perfil`, `nsfw_ui` | Ajustes de idioma, etiquetas y vista. |

---

## 5. Qué guarda `data/`

```
data/
├── config.json            # configuración (ver arriba)
├── known_people.json      # personas conocidas + vectores faciales
├── posts.json             # lo que se escribió por foto: pie, comentarios,
│                          #   favoritos, personas... (0 = no generado)
├── copetes.json           # titulares generados (📰) para la portada
├── tags.json              # etiquetas propias por foto
├── moldes.json            # tus moldes de pie de foto (editor de Pies)
├── phrases.json           # frases para la portada / secciones
├── person_meta.json       # metadata por persona
├── not_faces.json         # recortes descartados como "no persona"
├── procesadas.json        # fotos ya procesadas (evita repetir)
├── cola.json              # cola de trabajos en disco (ver sección 3)
├── deshacer.json          # última acción deshacible (mover/renombrar)
├── stats.json             # histórico de aciertos de sugerencias
├── clusters.json          # grupos del último análisis (se regenera)
├── labels.json            # nombres puestos en el análisis (se regenera)
├── duplicates.json        # resultado del detector de duplicados
├── gallery_thumbs/        # caché de miniaturas (grilla, home)
├── known_thumbs/          # portadas de personas conocidas
├── correct_thumbs/        # caché del corrector
├── dupe_thumbs/, find_thumbs/, nsfw_thumbs/, split_thumbs/, texto_thumbs/
└── backups/               # respaldos automáticos (si activás respaldo.py)
```

Los `.json` se guardan a medida que se genera cada foto (nunca se pierde lo ya
hecho). Las `*_thumbs/` son caché: se regeneran solas si las borrás.

---

## 6. Problemas comunes

| Síntoma | Causa / solución |
| --- | --- |
| "Veo la versión vieja" | Proceso fantasma en el puerto 5000. Cerrá el lanzador y volvé a abrirlo (mata el fantasma), o **Ctrl+F5** para el CSS/JS cacheado. |
| El ✨/🔮 no genera nada | Ollama caído (`http://127.0.0.1:11434`) o modelo no instalado. La web avisa con un cartel chico; `ollama list` para chequear. |
| Generación muy lenta | En CPU es normal (20–40 s por foto); con GPU, verificá que `KEEP_ALIVE=-1` y que el modelo quede cargado. |
| "Pijas"/textos inventados del modelo | Usá los modelos abliterated 🔞 y el diccionario de `f_conversiones`. |
| Está feo / se corta el modal del creador | CSS se sirve con hash `?v=` (auto): si editas CSS, la URL cambia sola. Un solo navegador cachea HTML: control-f5. |

---

## 7. Respaldos

Los snapshots manuales se guardan en `F:\projects\photo-sorter\backups\`
(cuando la app corre desde F:). La carpeta de datos completa
(`app` + `data`) se comprime a un zip con fecha al cierre de cada sesión de
trabajo. `app/respaldo.py` también tiene lógica de respaldo automático.