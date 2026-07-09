@echo off
rem Lanzador de photo-sorter: analiza las fotos y abre la web para etiquetar.
cd /d "%~dp0"

if not exist venv\Scripts\python.exe (
    echo Primero hay que instalar: hace doble clic en instalar.bat
    pause
    exit /b 1
)

if not exist data\photos mkdir data\photos

rem Contar fotos en data\photos
set COUNT=0
for /r data\photos %%f in (*.jpg *.jpeg *.png *.bmp *.webp) do set /a COUNT+=1
if %COUNT%==0 (
    echo No hay fotos todavia. Copia tus fotos a la carpeta data\photos
    echo ^(se abre ahora^) y volve a hacer doble clic en Photo Sorter.
    start "" explorer "%~dp0data\photos"
    pause
    exit /b 0
)

echo ============================================
echo  photo-sorter - %COUNT% foto(s) encontradas
echo ============================================
echo.
echo Paso 1/2: analizando rostros (puede tardar)...
venv\Scripts\python -m app.analyze || (pause & exit /b 1)

echo.
echo Paso 2/2: abriendo la web para poner nombres...
echo (Para terminar, cerra esta ventana)
venv\Scripts\python -m app.flask_app
pause
