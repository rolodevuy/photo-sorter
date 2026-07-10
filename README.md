# photo-sorter

Organizador de fotos por reconocimiento facial, **100% local**. Elegís una carpeta de **origen** (donde están tus fotos — no se mueven ni se modifican) y una de **destino**; el sistema detecta todos los rostros, agrupa automáticamente los de la misma persona, te pregunta en una web local "¿quién es esta persona?" y copia las fotos al destino en carpetas por nombre.

## Flujo (todo desde la web)

```
PhotoSorter.bat  →  se abre http://127.0.0.1:5000
```

1. **Elegir carpetas**: origen (tus fotos) y destino (donde se crean las carpetas por persona), con el selector de carpetas de Windows.
2. **Analizar**: detecta rostros y agrupa las caras iguales (clustering, sin entrenamiento). Muestra barra de progreso.
3. **Etiquetar**: cada grupo aparece con la pregunta "¿quién es esta persona?". Los grupos sin nombre se ignoran.
4. **Organizar**: copia cada foto a `<destino>\<nombre>\`. Una foto con varias personas queda en la carpeta de cada una. **Los originales no se tocan.**

## Buscar duplicados

Además de ordenar por rostro, la web tiene una pestaña **"Buscar duplicados"** que encuentra fotos repetidas en la carpeta de origen:

- **Exactas**: el mismo archivo aunque tenga otro nombre (comparación por contenido, SHA-1).
- **Parecidas**: la misma imagen reescalada, recomprimida o con marca de agua (hash perceptual *dHash*).

En cada grupo marca cuál conviene **conservar** (la de mayor resolución y peso). Al resolver, las sobrantes que tildes se **mueven** a una carpeta `_duplicados` dentro del origen — no se borran, así las revisás antes de eliminarlas. Por consola: `python -m app.duplicates`.

## Renombrar (nomenclador)

La pestaña **"Renombrar"** cambia en masa los nombres de las imágenes de una carpeta con el patrón que elijas: **palabra + separador + número** con la cantidad de dígitos que quieras (ej. `vacaciones_0001.jpg`, `IMG001.jpg`). Podés elegir el número inicial y ordenar por nombre o por fecha. Muestra una **vista previa** antes de aplicar, y renombra en dos pasos para no pisar archivos aunque el patrón nuevo coincida con nombres existentes. Trabaja solo en el primer nivel de la carpeta.

## Formatos de imagen soportados

`.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp` y `.gif` (los GIF, por su primer fotograma).

> **Motor de detección:** usa **YuNet** (detección) + **SFace** (reconocimiento) de OpenCV, que detectan bien caras de perfil o anguladas y son rápidos en CPU (~1 s por foto). Los modelos están en `app/models/` y corren 100% local.

## Privacidad

- Todo el reconocimiento corre en tu máquina (OpenCV YuNet+SFace, modelos incluidos en `app/models/`).
- La web solo escucha en `127.0.0.1`: nadie más en la red puede acceder.
- Ninguna foto ni dato sale de tu computadora. El código no hace ninguna conexión a internet.

## Archivos internos (carpeta `data/`)

```
data/
├── config.json        # carpetas de origen y destino elegidas
├── faces/             # miniaturas de rostros (para la web)
├── clusters.json      # grupos de rostros del último análisis
└── labels.json        # nombres puestos en la web
```

## Formatos soportados

`.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp` y `.gif`. Los GIF se analizan por su **primer fotograma** (si están animados, se ignora el resto). Archivos con la extensión equivocada (p. ej. un PNG guardado como `.jpg`) igual se leen bien: el formato se detecta por el contenido, no por el nombre.

## Ajustes

En `app/analyze.py`, la constante `EPS` (por defecto `0.45`) controla qué tan parecidos deben ser dos rostros para caer en el mismo grupo:

- **Bajarla** (ej. `0.40`) → grupos más estrictos: menos mezclas, pero una misma persona puede partirse en varios grupos (podés ponerle el mismo nombre a todos y se unifican al organizar).
- **Subirla** (ej. `0.50`) → más permisivo: riesgo de mezclar personas parecidas.

Ver [SETUP.md](SETUP.md) para la instalación paso a paso.
