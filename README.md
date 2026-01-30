# AINex Soccer – MuJoCo Simulation & Action Groups

This repository contains a MuJoCo simulation of the AINex humanoid robot,
along with extracted hardware action groups converted into CSV format
for analysis, replay, and learning-based control.

## Structure

- assets/ainex/  
  Robot meshes and MJCF/URDF files
- assets/action_groups/raw/  
  Original hardware action group (.d6a SQLite databases)
- assets/action_groups/csv/  
  CSV exports of action groups (Servo1–Servo22 + timing)
- scripts/  
  Viewers and replay tools
- tools/  
  Conversion and export utilities

## Quick Start

```bash
pip install mujoco numpy
mjpython scripts/view_ainex_stable.py
```
