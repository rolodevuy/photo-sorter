# photo-sorter

Organizador de fotos por reconocimiento facial, **100% local**. No necesita entrenamiento previo ni conexión a internet: detecta todos los rostros de tus fotos, agrupa automáticamente los que son de la misma persona, te pregunta en una web local "¿quién es esta persona?" y guarda las fotos en carpetas por nombre.

## Flujo

1. **Analizar** (`python -m app.analyze`): recorre `data/photos/`, detecta rostros, calcula sus huellas faciales y agrupa las parecidas (clustering, sin entrenamiento). Genera miniaturas y `data/clusters.json`.
2. **Etiquetar** (`python -m app.flask_app`): abre una web en `http://127.0.0.1:5000` que muestra cada grupo de rostros y pregunta quién es. Los nombres quedan en `data/labels.json`.
3. **Organizar** (botón en la web, o `python -m app.organizer`): copia cada foto a `data/sorted/<nombre>/`. Una foto con varias personas queda en la carpeta de cada una. **Los originales no se tocan.**

## Privacidad

- Todo el reconocimiento corre en tu máquina (dlib/face_recognition, modelos incluidos en la instalación).
- La web solo escucha en `127.0.0.1`: nadie más en la red puede acceder.
- Ninguna foto ni dato sale de tu computadora. El código no hace ninguna conexión a internet.

## Estructura de datos

```
data/
├── photos/            # ENTRADA: tus fotos (se aceptan subcarpetas)
├── faces/             # miniaturas de rostros (generado)
├── clusters.json      # grupos de rostros (generado)
├── labels.json        # nombres que pusiste en la web (generado)
└── sorted/            # SALIDA: carpetas por persona
    ├── mamá/
    └── juan/
```

## Ajustes

En `app/analyze.py`, la constante `EPS` (por defecto `0.45`) controla qué tan parecidos deben ser dos rostros para caer en el mismo grupo:

- **Bajarla** (ej. `0.40`) → grupos más estrictos: menos mezclas, pero una misma persona puede partirse en varios grupos (podés ponerle el mismo nombre a todos y se unifican al organizar).
- **Subirla** (ej. `0.50`) → más permisivo: riesgo de mezclar personas parecidas.

Ver [SETUP.md](SETUP.md) para la instalación paso a paso.
