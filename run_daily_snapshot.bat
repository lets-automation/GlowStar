@echo off
REM Daily GlowStar snapshot job (brief Section 3.5). Banks an immutable, dated
REM inventory+sales snapshot AND the live Master grid. Registered as the Windows
REM task "GlowStarSnapshot" (daily 02:00, StartWhenAvailable). Safe to run by hand.
cd /d "%~dp0"
".venv\Scripts\python.exe" -m glowstar.ingestion.run_snapshot >> "data\snapshots\snapshot_job.log" 2>&1
exit /b %ERRORLEVEL%
