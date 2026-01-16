#!/bin/bash
gnome-terminal -- bash -c "
echo nvidia|sudo -S chmod +777 /dev/ttyTHS0
cd Documents/BJTU_Under_Water_Visual_System/lateral_docking
uv run lateral_docking/src/main.py
"