@echo off
REM Nightly GlowStar retrain. Rebuilds records.json FRESH from the live API
REM (current stock + latest sales + live BGM), retrains, and promotes the model
REM ONLY if it passes the accuracy gate (a bad data day can't degrade live pricing).
REM Registered as the Windows task "GlowStarRetrain" (daily 03:00, StartWhenAvailable).
cd /d "%~dp0"
".venv\Scripts\python.exe" -m glowstar.training.retrain >> "data\snapshots\retrain_job.log" 2>&1
exit /b %ERRORLEVEL%
