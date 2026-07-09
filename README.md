# photo-sorter

Organizador de fotos por reconocimiento facial. Entrenás con fotos de referencia de cada persona y después el sistema detecta quién aparece en tus fotos nuevas para poder clasificarlas.

## Cómo funciona

1. **Indexer** (`app/indexer.py`): recorre `data/training/<nombre_persona>/` con fotos de referencia, calcula los *encodings* faciales y los guarda en `data/encodings.pkl`.
2. **Detector** (`app/detector.py`): recorre `data/photos/` con fotos nuevas, detecta rostros, los compara contra los encodings entrenados y escribe los resultados en `data/detections.json`.
3. **Organizer** (`app/organizer.py`): *(pendiente)* mueve/copia las fotos a carpetas por persona según las detecciones.
4. **Interfaz web** (`app/flask_app.py`): *(pendiente)* revisión y corrección manual de detecciones.

## Estructura de datos

```
data/
├── training/
│   ├── juan/          # fotos de referencia de Juan
│   └── maria/         # fotos de referencia de María
├── photos/            # fotos nuevas a clasificar
├── encodings.pkl      # generado por el indexer
└── detections.json    # generado por el detector
```

## Uso rápido

```bash
python -m app.indexer     # entrenar con fotos de referencia
python -m app.detector    # detectar rostros en fotos nuevas
```

Ver [SETUP.md](SETUP.md) para la instalación paso a paso.
