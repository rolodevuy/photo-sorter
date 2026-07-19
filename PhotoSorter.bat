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

rem Si quedo un servidor viejo ocupando el puerto 5000 (una ventana anterior,
rem un proceso colgado), echarlo. Sin esto, el arranque fallaba en silencio y
rem el navegador seguia mostrando la version vieja: parecia que "los cambios
rem no aparecian" cuando en realidad corria otro proceso.
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":5000" ^| findstr "LISTENING"') do taskkill /F /PID %%a >nul 2>&1

echo Abriendo la web en el navegador...
echo (Para terminar el programa, cerra esta ventana)
echo.
venv\Scripts\python -m app.flask_app
pause
