@echo off
setlocal
cd /d "%~dp0"

set "PYTHON_ENV=.venv\Scripts\python.exe"

if exist "%PYTHON_ENV%" goto ejecutar

echo [1/3] Buscando Python 3.11, 3.12 o 3.13...
set "PY_CMD="

where py >nul 2>&1
if not errorlevel 1 (
    py -3.13 -c "import sys" >nul 2>&1
    if not errorlevel 1 set "PY_CMD=py -3.13"
    if not defined PY_CMD (
        py -3.12 -c "import sys" >nul 2>&1
        if not errorlevel 1 set "PY_CMD=py -3.12"
    )
    if not defined PY_CMD (
        py -3.11 -c "import sys" >nul 2>&1
        if not errorlevel 1 set "PY_CMD=py -3.11"
    )
)

if not defined PY_CMD (
    where python >nul 2>&1
    if not errorlevel 1 (
        python -c "import sys; raise SystemExit(0 if (3, 11) <= sys.version_info[:2] < (3, 14) else 1)" >nul 2>&1
        if not errorlevel 1 set "PY_CMD=python"
    )
)

if not defined PY_CMD (
    echo.
    echo No se encontro Python 3.11, 3.12 o 3.13.
    echo Instala Python desde https://www.python.org/downloads/ y vuelve a ejecutar este archivo.
    pause
    exit /b 1
)

echo [2/3] Creando entorno virtual...
%PY_CMD% -m venv .venv
if errorlevel 1 goto error

echo [3/3] Instalando dependencias de la interfaz...
"%PYTHON_ENV%" -m pip install --upgrade pip
if errorlevel 1 goto error
"%PYTHON_ENV%" -m pip install -r requirements-runtime.txt
if errorlevel 1 goto error

:ejecutar
if not exist "models\hybrid\sign_hybrid.pt" (
    echo No se encontro models\hybrid\sign_hybrid.pt
    pause
    exit /b 1
)

echo Abriendo reconocimiento de senas...
"%PYTHON_ENV%" scripts\live_gui.py --cnn-model models\hybrid\sign_hybrid.pt --backend auto
if errorlevel 1 goto error
exit /b 0

:error
echo.
echo No se pudo preparar o ejecutar la aplicacion.
echo Revisa el mensaje anterior y tu conexion a Internet.
pause
exit /b 1
