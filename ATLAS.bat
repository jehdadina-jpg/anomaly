@echo off
title Anomaly Terminal Server
color 0A

echo =======================================================
echo          STARTING ANOMALY DETECTION TERMINAL           
echo =======================================================
echo.
echo The terminal server is booting up...
echo It will fetch the latest market quotes and then automatically 
echo open your default web browser to the terminal interface.
echo.
echo IMPORTANT: Leave this black window open while using the terminal!
echo Closing this window will shut down the live data feed.
echo.

:: Check if python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python was not found on your system!
    echo Please make sure Python is installed and added to your system PATH.
    pause
    exit /b
)

:: Run the backend server
python api.py

:: If the server crashes or closes, pause so the user can read the error
echo.
echo The server has stopped.
pause
