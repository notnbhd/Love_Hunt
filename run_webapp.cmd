@echo off
title Dating Recommendation System
echo ============================================================
echo        DATING RECOMMENDATION SYSTEM
echo ============================================================
echo.
echo Starting webapp... (opens in browser automatically)
echo.

cd /d "%~dp0"
call conda activate dating
python -m streamlit run src/app.py --server.headless true

pause
