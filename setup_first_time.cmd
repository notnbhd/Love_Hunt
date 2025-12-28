@echo off
title Dating Recommendation System - First Time Setup
echo ============================================================
echo    DATING RECOMMENDATION SYSTEM - FIRST TIME SETUP
echo ============================================================
echo.
echo This will pre-compute all models (only needed ONCE).
echo After this, the webapp will load instantly!
echo.
pause

cd /d "%~dp0"
call conda activate dating
python precompute.py

echo.
echo ============================================================
echo    SETUP COMPLETE!
echo ============================================================
echo.
echo You can now use 'run_webapp.cmd' to launch the app instantly.
echo.
pause
