@echo off
rem Instalador de photo-sorter: crea el entorno e instala las dependencias.
rem Se corre UNA sola vez (necesita internet solo esta vez).
cd /d "%~dp0"

echo ============================================
echo  photo-sorter - Instalacion
echo ============================================
echo.

if not exist venv (
    echo Creando entorno virtual...
    python -m venv venv || goto :error
)

echo Instalando dependencias (puede tardar unos minutos, sobre todo torch)...
venv\Scripts\python -m pip install --upgrade pip || goto :error
venv\Scripts\pip install -r requirements.txt || goto :error

echo.
echo ============================================
echo  Listo! Instalacion completa.
echo  Hace doble clic en "Photo Sorter" (escritorio).
echo  Para la IA (pie, etiquetas, vision, copete)
echo  hace falta Ollama con los modelos: ver SETUP.md.
echo ============================================
pause
exit /b 0

:error
echo.
echo Hubo un error en la instalacion. Revisa el mensaje de arriba.
pause
exit /b 1
