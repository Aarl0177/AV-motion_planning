# Autonomous Vehicle NMPC Simulation in MuJoCo

This project demonstrates a nonlinear model predictive control (NMPC) framework for an autonomous vehicle using a kinematic bicycle model. The vehicle tracks a reference path while avoiding dynamic obstacles. MuJoCo is used as a 3D visualization and replay environment.

## Features

- Kinematic bicycle model
- Nonlinear model predictive control using CasADi
- Dynamic obstacle prediction
- Obstacle avoidance with safety constraints
- Car MuJoCo visualization

## Simulation Overview

The vehicle state is:

```math
x = [p_x, p_y, \psi, v]
