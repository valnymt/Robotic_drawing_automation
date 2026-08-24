# draw-arm

A 2-DOF planar RR robot arm that takes a photo, extracts line art, and
draws it as a sequence of pen strokes — simulated in PyBullet, with a
live dashboard to watch it work. Theory reference: see
`robotics_theory_reference.md` (DH parameters, inverse kinematics,
trajectory generation, vision pipeline, coordinate transforms).

## Setup

```bash
pip install -r requirements.txt
```

Shared constants (link lengths, workspace bounds, vision thresholds)
live in `config.py` — every script imports from there rather than
redefining numbers.

## Phases

| # | Phase | Files |
|---|---|---|
| 0 | Scaffold & config | `requirements.txt`, `config.py`, `draw_arm.urdf` |
| 1 | Photo -> line art (Canny, contours, greedy stroke ordering) | `01_make_lineart.py` |
| 2 | Kinematics (FK/IK/reachability) | `kinematics.py` |
| 3 | Image space -> robot workspace mapping | `mapping.py` |
| 4 | Trajectory generation (joint-space interpolation) | `trajectory.py` |
| 5 | PyBullet drawing simulation -> MP4 | `02_simulate_draw.py` |
| 6 | Top-down verification render | `03_verify_topdown.py` |
| 7 | Live robot-vision dashboard (WebSocket) | `04_live_dashboard.py` |
| 8 | End-to-end CLI & error handling | `run.py` |
| 9 | Real hardware: dynamics & PID (future) | `dynamics.py`, `pid_controller.py`, `serial_driver.py` |

Phases 1-8 are built and run in order; each later phase assumes the
files from earlier phases already exist.
