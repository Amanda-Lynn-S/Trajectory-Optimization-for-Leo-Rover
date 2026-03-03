# Rover Trajectory Optimization via Sequential Convex Programming (SCP)

This project implements a **2D rover trajectory optimization framework** using **Sequential Convex Programming (SCP)** with **collision avoidance via Signed Distance Fields (SDFs)**.  
It follows the methodology of “GuSTO: Guaranteed Sequential Trajectory Optimization via Sequential Convex Programming” (Bonalli et al., ICRA 2019) for safe, smooth, and dynamically consistent path planning.
The codebase is inspired by the following reference: https://github.com/ynakka/SCPpy.

---

# Project Overview

The project consists of four main components:

| Module | Description |
|---------|-------------|
| **`SDF_Grid_Map_Gen.py`** | Graphical tool (Tkinter GUI) for creating and saving a **2D occupancy grid map**, defining start/end rover positions along with obstacles, and exporting a **Signed Distance Field (SDF)** as a CSV and PNG. |
| **`rover_model.py`** | Defines the rover’s **continuous-time dynamics**, **Taylor-series linearization**, and **exact zero-order-hold (ZOH) discretization**. |
| **`Rover.py`** | Defines the **Rover class**, combining model and constraints data (state and control limits, map data, etc.), and interpolation setup to get the sdf values and gradients corresponding to the rover location. Provides an interface used by the SCP solver. |
| **`SCP.py`** | Implements the **Sequential Convex Programming algorithm** that solves a convex subproblem iteratively until convergence. Handles dynamics linearization and discretization, constraints, trust-region updates, and cost minimization. |
| **`rover_pp_simulation.py`** | The **main simulation script** that sets up the problem, runs SCP, saves results, and visualizes the optimized trajectory. |

---
