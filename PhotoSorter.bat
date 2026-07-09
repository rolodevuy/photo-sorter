@echo off
rem Lanzador de photo-sorter: abre la web local donde se maneja todo.
cd /d "%~dp0"

if not exist venv\Scripts\python.exe (
    echo Primero hay que instalar: hace doble clic en instalar.bat
    pause
    exit /b 1
)

echo ============================================
echo  photo-sorter
echo ============================================
echo Abriendo la web en el navegador...
echo (Para terminar el programa, cerra esta ventana)
echo.
venv\Scripts\python -m app.flask_app
pause
