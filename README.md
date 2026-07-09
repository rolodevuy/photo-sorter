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

## Privacidad

- Todo el reconocimiento corre en tu máquina (dlib/face_recognition, modelos incluidos en la instalación).
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

## Ajustes

En `app/analyze.py`, la constante `EPS` (por defecto `0.45`) controla qué tan parecidos deben ser dos rostros para caer en el mismo grupo:

- **Bajarla** (ej. `0.40`) → grupos más estrictos: menos mezclas, pero una misma persona puede partirse en varios grupos (podés ponerle el mismo nombre a todos y se unifican al organizar).
- **Subirla** (ej. `0.50`) → más permisivo: riesgo de mezclar personas parecidas.

Ver [SETUP.md](SETUP.md) para la instalación paso a paso.
