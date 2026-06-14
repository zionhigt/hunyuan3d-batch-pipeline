@echo off
REM Launcher for the Hunyuan3D batch pipeline on Windows.
REM Activates the conda environment "hy3d" and runs src/batch.py.
REM
REM Usage: run_batch.bat [extra args passed to batch.py]

setlocal

set "PROJECT_ROOT=%~dp0.."
cd /d "%PROJECT_ROOT%"

call conda activate hy3d
if errorlevel 1 (
    echo [ERROR] Could not activate conda environment "hy3d".
    echo Make sure Miniconda is installed and the env was created:
    echo   conda create -n hy3d python=3.10 -y
    exit /b 1
)

python src\batch.py %*

endlocal
