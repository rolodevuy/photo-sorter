# Guía de instalación y uso

## 1. Crear el entorno

```bash
cd photo-sorter
python -m venv venv
venv\Scripts\activate
```

## 2. Instalar dependencias (Windows)

`face-recognition` depende de **dlib**, que necesita compilarse. Hay dos caminos:

**Opción A (recomendada): wheel precompilado**

```bash
pip install dlib-bin
pip install face-recognition --no-deps
pip install face-recognition-models numpy scikit-learn Pillow Flask
```

**Opción B: compilar dlib**

Instalar primero [CMake](https://cmake.org/download/) y las *Build Tools for Visual Studio* (carga de trabajo "Desarrollo para el escritorio con C++"), y después:

```bash
pip install -r requirements.txt
```

> Los modelos de reconocimiento se instalan junto con las librerías. Después de este paso **no se necesita internet para nada**.

## 3. Poner las fotos

Copiá las fotos que querés clasificar en `data/photos/` (se aceptan subcarpetas):

```
data/photos/IMG_001.jpg
data/photos/vacaciones/IMG_045.jpg
...
```

## 4. Analizar

```bash
python -m app.analyze
```

Detecta todos los rostros y arma los grupos. Con muchas fotos puede tardar (aprox. 1–3 segundos por foto en CPU).

## 5. Ponerle nombre a cada persona

```bash
python -m app.flask_app
```

Abrí **http://127.0.0.1:5000** en el navegador. Vas a ver cada grupo de rostros con la pregunta "¿quién es esta persona?". Escribí el nombre y tocá Guardar. Los grupos que no te interesan los podés dejar sin nombre.

- Clic en una miniatura → abre la foto original completa.
- Si una misma persona aparece en dos grupos, ponéles el mismo nombre: se unifican al organizar.

## 6. Organizar

En la misma web, tocá el botón **"Organizar fotos"** (o corré `python -m app.organizer`). Cada foto se **copia** a `data/sorted/<nombre>/`. Los originales de `data/photos/` quedan intactos.

## Si agregás fotos nuevas

Volvé a correr `python -m app.analyze` y después la web. (Por ahora el análisis rehace todo desde cero; los nombres guardados se pierden al re-analizar — mejora pendiente.)
