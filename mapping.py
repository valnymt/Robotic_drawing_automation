"""Image pixel space -> robot workspace coordinates.

See robotics_theory_reference.md section 9. Image row increases
downward; robot y increases upward, hence the (H - row) flip.
"""

import logging

import numpy as np

import kinematics
from config import X0, X1, Y0, Y1

logger = logging.getLogger(__name__)


def img_to_arm(col: float, row: float, w: int, h: int) -> tuple[float, float]:
    x = X0 + (col / w) * (X1 - X0)
    y = Y0 + ((h - row) / h) * (Y1 - Y0)
    return x, y


def strokes_to_arm(strokes_px: list, w: int, h: int) -> list[np.ndarray]:
    """Map a list of pixel-space polylines to arm-space, dropping any
    point outside the reachable ring (with a logged warning) rather than
    letting an unreachable IK call blow up downstream."""
    strokes_arm = []
    dropped = 0

    for stroke in strokes_px:
        mapped = []
        for col, row in stroke:
            x, y = img_to_arm(col, row, w, h)
            if kinematics.is_reachable(x, y):
                mapped.append((x, y))
            else:
                dropped += 1
                logger.warning(
                    "dropping unreachable point: pixel=(%s, %s) -> arm=(%.4f, %.4f)",
                    col, row, x, y,
                )
        if len(mapped) >= 2:
            strokes_arm.append(np.array(mapped, dtype=np.float64))

    if dropped:
        logger.warning("dropped %d unreachable point(s) total", dropped)

    return strokes_arm
