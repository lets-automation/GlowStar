@echo off
REM Nightly GlowStar retrain. Rebuilds records.json FRESH from the live API
REM (current stock + latest sales + live BGM), retrains, and promotes the model
REM ONLY if it passes the accuracy gate (a bad data day can't degrade live pricing).
REM Registered as the Windows task "GlowStarRetrain" (daily 03:00, StartWhenAvailable).
cd /d "%~dp0"
".venv\Scripts\python.exe" -m glowstar.training.retrain >> "data\snapshots\retrain_job.log" 2>&1
set PRICING_RC=%ERRORLEVEL%

REM Workstream B: the velocity model behind tradeability, bifurcation and the
REM inventory reports. Its own registry and its own promotion gate (C-index AND
REM calibration, both against the segment-median baseline the FrontOffice field
REM ships today). Runs even if the pricing retrain failed -- they are independent
REM models and a bad pricing data day should not also freeze velocity.
".venv\Scripts\python.exe" -m glowstar.training.velocity_retrain >> "data\snapshots\retrain_job.log" 2>&1
set VELOCITY_RC=%ERRORLEVEL%

if not "%PRICING_RC%"=="0" exit /b %PRICING_RC%
exit /b %VELOCITY_RC%
