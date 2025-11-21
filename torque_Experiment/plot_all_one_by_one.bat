@echo off
echo Plotting PID...
python -m src.plot_controller --which pid
echo.

echo Plotting BC...
python -m src.plot_controller --which bc
echo.

echo Plotting RL...
python -m src.plot_controller --which rl
echo.

echo Plotting ALL together...
python -m src.plot_controller --which all

echo Done plotting all graphs in sequence!
pause
