@echo off
rem Launches (or relaunches) the post-round-1 experiment cycle inside WSL (see docs/UPGRADE_PLAN.md).
rem Edit the two "cd /mnt/e/<repository root>" paths below to the WSL path of your clone before use.
rem The cycle runs in its own minimized WSL window ("worker - do not close"); this window only tails the log and may be closed.
rem If runs\cycle\KILL exists, the running cycle and its workers are stopped first (the file is then removed).
title SELF-EVOLVE-SEARCH cycle log
powercfg /change standby-timeout-ac 0 >nul 2>&1
powercfg /change hibernate-timeout-ac 0 >nul 2>&1
if exist "%~dp0..\runs\cycle\KILL" wsl.exe -- bash -lc "cd /mnt/e/<repository root> && if [ -f runs/cycle/KILL ]; then echo stopping the running cycle; [ -f runs/cycle/cycle.pid ] && pkill -TERM -P $(cat runs/cycle/cycle.pid) 2>/dev/null; [ -f runs/cycle/cycle.pid ] && kill -TERM $(cat runs/cycle/cycle.pid) 2>/dev/null; for s in memorization_check.py train_update.py collect_batch.py evaluate.py run_experiment.py; do pkill -TERM -f scripts/$s 2>/dev/null; done; sleep 5; for s in memorization_check.py train_update.py run_cycle.sh; do pkill -KILL -f scripts/$s 2>/dev/null; done; .venv-data/bin/python scripts/stop_server.py 2>/dev/null; rm -f runs/cycle/KILL; echo stopped; fi"
wsl.exe -- bash -lc "cd /mnt/e/<repository root> && mkdir -p runs/cycle && if [ -f runs/cycle/cycle.pid ] && kill -0 $(cat runs/cycle/cycle.pid) 2>/dev/null; then echo cycle already running; exit 0; else exit 1; fi"
if errorlevel 1 start "SELF-EVOLVE-SEARCH worker - do not close" /min wsl.exe -- bash -lc "cd /mnt/e/<repository root> && exec bash scripts/run_cycle.sh > runs/cycle-main.log 2>&1"
wsl.exe -- bash -lc "cd /mnt/e/<repository root> && sleep 4 && nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv 2>/dev/null; tail -n 30 -f runs/cycle/cycle.log"
pause
