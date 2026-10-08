@echo off
rem Launches (or relaunches) cycle 2 (seeds, baselines, final benchmark; see scripts/run_cycle2.sh) inside WSL.
rem The cycle runs in its own minimized WSL window ("worker - do not close"); this window only tails the log and may be closed.
rem If runs\cycle2\KILL exists, the running cycle and its workers are stopped first (the file is then removed).
rem Edit the "cd /mnt/e/<repository root>" paths below to the WSL path of your clone before use.
title SELF-EVOLVE-SEARCH cycle 2 log
powercfg /change standby-timeout-ac 0 >nul 2>&1
powercfg /change hibernate-timeout-ac 0 >nul 2>&1
if exist "%~dp0..\runs\cycle2\KILL" wsl.exe -- bash -lc "cd /mnt/e/<repository root> && if [ -f runs/cycle2/KILL ]; then echo stopping the running cycle; [ -f runs/cycle2/cycle.pid ] && pkill -TERM -P $(cat runs/cycle2/cycle.pid) 2>/dev/null; [ -f runs/cycle2/cycle.pid ] && kill -TERM $(cat runs/cycle2/cycle.pid) 2>/dev/null; for s in memorization_check.py train_update.py collect_batch.py evaluate.py run_experiment.py; do pkill -TERM -f scripts/$s 2>/dev/null; done; sleep 5; for s in memorization_check.py train_update.py run_cycle.sh run_cycle2.sh; do pkill -KILL -f scripts/$s 2>/dev/null; done; .venv-data/bin/python scripts/stop_server.py 2>/dev/null; rm -f runs/cycle2/KILL; echo stopped; fi"
wsl.exe -- bash -lc "cd /mnt/e/<repository root> && mkdir -p runs/cycle2 && if [ -f runs/cycle2/cycle.pid ] && kill -0 $(cat runs/cycle2/cycle.pid) 2>/dev/null; then echo cycle already running; exit 0; else exit 1; fi"
if errorlevel 1 start "SELF-EVOLVE-SEARCH cycle 2 worker - do not close" /min wsl.exe -- bash -lc "cd /mnt/e/<repository root> && exec bash scripts/run_cycle2.sh > runs/cycle2-main.log 2>&1"
wsl.exe -- bash -lc "cd /mnt/e/<repository root> && sleep 4 && nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv 2>/dev/null; tail -n 30 -f runs/cycle2/cycle.log"
pause
