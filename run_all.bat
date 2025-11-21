@echo off
REM ------------------------------
REM run_all.bat - Hybrid IL + RL pipeline
REM Place this file in project root and run it from Command Prompt
REM ------------------------------

REM 1) Activate venv (works for Command Prompt)
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
) else (
    echo Virtual environment not found at .venv\Scripts\activate.bat
    echo Make sure you created the venv in the project root.
    pause
    exit /b 1
)

REM ensure outputs folder exists
if not exist outputs (
    mkdir outputs
)

echo.
echo ===============================
echo HYBRID IL + RL FULL PIPELINE
echo ===============================
echo.

echo [1/4] Collecting expert data...
python -m src.collect_expert --episodes 120 --out outputs/expert_data.npz
if ERRORLEVEL 1 (
    echo Collect expert failed. Aborting.
    pause
    exit /b 1
)

echo [2/4] Training behavioral cloning model...
python -m src.imitation.behavioral_cloning --data outputs/expert_data.npz --save outputs/bc_policy.pth --epochs 100
if ERRORLEVEL 1 (
    echo Behavioral cloning failed. Aborting.
    pause
    exit /b 1
)

echo [3/4] Training PPO with BC initialization...
python -m src.rl.train_rl --timesteps 200000 --bc outputs/bc_policy.pth --out outputs/rl_model.zip
if ERRORLEVEL 1 (
    echo RL training failed. Aborting.
    pause
    exit /b 1
)

echo [4/4] Evaluating final model...
python -m src.evaluate.evaluate_policy --model outputs/rl_model.zip --episodes 50
if ERRORLEVEL 1 (
    echo Evaluation failed.
    pause
    exit /b 1
)

echo.
echo ===============================
echo      Pipeline complete!
echo ===============================
pause
