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

echo Instalando dependencias (puede tardar unos minutos)...
venv\Scripts\python -m pip install --upgrade pip || goto :error
venv\Scripts\pip install opencv-contrib-python numpy scikit-learn Pillow Flask || goto :error

echo.
echo ============================================
echo  Listo! Instalacion completa.
echo  Hace doble clic en "Photo Sorter" (escritorio)
echo  y elegi las carpetas desde la web.
echo ============================================
pause
exit /b 0

:error
echo.
echo Hubo un error en la instalacion. Revisa el mensaje de arriba.
pause
exit /b 1
