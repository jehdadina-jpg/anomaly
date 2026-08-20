@echo off
title ATLAS - AI Trading and Analysis System
cls
echo ========================================
echo    ATLAS - AI Trading Analysis System
echo ========================================
echo.

cd /d "%~dp0"

REM Build the data panel if missing (joins OHLCV with NSE bhavcopy)
if not exist "data\features\panel.csv" (
    echo [SETUP] Building data panel...
    python -m data.build_panel
    if errorlevel 1 (
        echo ERROR: Panel build failed
        pause
        exit /b 1
    )
    echo.
)

REM Train the prediction model if missing
if not exist "results\models\prediction\lgbm_ensemble.pkl" (
    echo [SETUP] First time setup - training prediction model...
    echo         Runs walk-forward validation; takes a few minutes.
    echo.
    python -m training.train
    if errorlevel 1 (
        echo ERROR: Model training failed
        pause
        exit /b 1
    )
    echo.
)

echo [LAUNCH] Starting ATLAS API server...
echo.
echo Access the terminal at: http://localhost:8001
echo Press Ctrl+C to stop the server
echo.
python api.py
