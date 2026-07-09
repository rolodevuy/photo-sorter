# Guía de instalación y uso

## 1. Requisitos previos (Windows)

`face-recognition` depende de **dlib**, que necesita compilarse. En Windows hay dos caminos:

**Opción A (recomendada): wheel precompilado**

```bash
pip install dlib-bin
pip install face-recognition --no-deps
pip install face-recognition-models numpy Pillow Flask
```

**Opción B: compilar dlib**

Instalar primero [CMake](https://cmake.org/download/) y las *Build Tools for Visual Studio* (carga de trabajo "Desarrollo para el escritorio con C++"), y después:

```bash
pip install -r requirements.txt
```

## 2. Instalación

```bash
git clone https://github.com/rolodevuy/photo-sorter.git
cd photo-sorter
python -m venv venv
venv\Scripts\activate
```

Luego instalar dependencias según la Opción A o B de arriba.

## 3. Entrenar con fotos de referencia

Crear una carpeta por persona dentro de `data/training/`, con 3–10 fotos claras de esa persona (idealmente con un solo rostro por foto):

```
data/training/juan/foto1.jpg
data/training/juan/foto2.jpg
data/training/maria/foto1.jpg
...
```

Ejecutar:

```bash
python -m app.indexer
```

Esto genera `data/encodings.pkl`.

## 4. Detectar rostros en fotos nuevas

Poner las fotos a clasificar en `data/photos/` y ejecutar:

```bash
python -m app.detector
```

Los resultados quedan en `data/detections.json`, con este formato:

```json
{
  "vacaciones/IMG_001.jpg": [
    {"name": "juan", "distance": 0.42, "box": {"top": 10, "right": 200, "bottom": 150, "left": 60}},
    {"name": "desconocido", "distance": 0.71, "box": {"top": 30, "right": 400, "bottom": 180, "left": 260}}
  ]
}
```

## 5. Ajustar la precisión

En `app/detector.py` está la constante `TOLERANCE` (por defecto `0.6`):

- **Bajarla** (ej. `0.5`) → más estricto: menos falsos positivos, pero más rostros quedan como "desconocido".
- **Subirla** (ej. `0.65`) → más permisivo: reconoce más, con riesgo de confundir personas.

## Próximos pasos

- Interfaz web para revisión/correcciones (`app/flask_app.py`)
- Organizer funcional (`app/organizer.py`)
