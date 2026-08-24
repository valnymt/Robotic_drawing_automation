"""Arm-space stroke list -> joint-angle time series.

Each stroke is drawn (pen_down=True) with joint-space interpolation
between its waypoints; the travel move from one stroke's end to the
next stroke's start is a pen-up move (pen_down=False). See
robotics_theory_reference.md section 6.
"""

from dataclasses import dataclass

import numpy as np

import kinematics
from config import STEPS_PER_SEGMENT


@dataclass
class Sample:
    t1: float
    t2: float
    pen_down: bool


def _linear_blend(a: float, b: float, frac: float) -> float:
    return a + (b - a) * frac


def _quintic_blend(a: float, b: float, frac: float) -> float:
    # Smoothstep-5: position/velocity/acceleration all pinned to 0 at
    # frac=0 and frac=1 (Euler-Lagrange quintic trajectory, section 6).
    s = 6 * frac**5 - 15 * frac**4 + 10 * frac**3
    return a + (b - a) * s


def _interp_segment(start: tuple[float, float], end: tuple[float, float],
                     steps: int, quintic: bool) -> list[tuple[float, float]]:
    blend = _quintic_blend if quintic else _linear_blend
    samples = []
    for i in range(1, steps + 1):
        frac = i / steps
        t1 = blend(start[0], end[0], frac)
        t2 = blend(start[1], end[1], frac)
        samples.append((t1, t2))
    return samples


def generate_trajectory(
    strokes_arm: list[np.ndarray],
    steps_per_segment: int = STEPS_PER_SEGMENT,
    quintic: bool = False,
) -> list[Sample]:
    """strokes_arm: list of Nx2 arrays of (x, y) arm-space waypoints,
    already ordered (e.g. by 01_make_lineart.py + mapping.py).

    Returns a flat list of Sample(t1, t2, pen_down).
    """
    if not strokes_arm:
        return []

    samples: list[Sample] = []
    current_joint = kinematics.ik(*strokes_arm[0][0])
    samples.append(Sample(current_joint[0], current_joint[1], pen_down=False))

    for stroke_idx, stroke in enumerate(strokes_arm):
        stroke_start_joint = kinematics.ik(*stroke[0])

        # Travel move (pen up) from wherever we are to this stroke's start.
        for t1, t2 in _interp_segment(current_joint, stroke_start_joint, steps_per_segment, quintic):
            samples.append(Sample(t1, t2, pen_down=False))
        current_joint = stroke_start_joint

        # Draw the stroke itself (pen down) through each waypoint.
        for point in stroke[1:]:
            waypoint_joint = kinematics.ik(*point)
            for t1, t2 in _interp_segment(current_joint, waypoint_joint, steps_per_segment, quintic):
                samples.append(Sample(t1, t2, pen_down=True))
            current_joint = waypoint_joint

    return samples
