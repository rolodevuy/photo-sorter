# Guía de instalación y uso

## 1. Instalar (una sola vez)

Doble clic en **`instalar.bat`**. Crea el entorno e instala las dependencias (necesita internet solo esta vez; puede tardar unos minutos).

<details>
<summary>Instalación manual (equivalente)</summary>

```bash
cd photo-sorter
python -m venv venv
venv\Scripts\pip install dlib-bin
venv\Scripts\pip install --no-deps face-recognition face-recognition-models
venv\Scripts\pip install numpy scikit-learn Pillow Flask click "setuptools<81"
```

Notas:
- `dlib-bin` es el wheel precompilado de dlib para Windows (evita compilar con CMake/Visual Studio).
- `setuptools<81` es necesario porque `face_recognition_models` usa `pkg_resources`, eliminado en setuptools 81+.
</details>

## 2. Usar

Doble clic en **`PhotoSorter.bat`** (o en el acceso directo "Photo Sorter"). Se abre la web en `http://127.0.0.1:5000` y ahí:

1. **Origen**: elegí la carpeta donde están tus fotos (se leen también las subcarpetas; no se mueven ni modifican).
2. **Destino**: elegí dónde querés que se creen las carpetas por persona.
3. **Guardar carpetas** → **🔍 Analizar fotos del origen**. Con muchas fotos tarda (~1–3 s por foto); hay barra de progreso.
4. Cuando termina, aparece cada grupo de rostros: escribí **quién es cada persona** y guardá. Clic en una miniatura abre la foto original. Si una misma persona quedó en dos grupos, ponéles el mismo nombre: se unifican al organizar.
5. **📁 Organizar** → cada foto se **copia** a `<destino>\<nombre>\`. Los originales quedan intactos.

Para cerrar el programa: cerrá la ventana negra.

## Si agregás fotos nuevas al origen

Volvé a tocar "Analizar". El análisis rehace todo desde cero, así que los nombres puestos antes se pierden y hay que ponerlos de nuevo (mejora pendiente: análisis incremental que recuerde a las personas).
