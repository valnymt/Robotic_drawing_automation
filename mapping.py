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


def strokes_to_arm(strokes_px: list, w: int, h: int, collect_dropped: bool = False):
    """Map a list of pixel-space polylines to arm-space, dropping any
    point outside the reachable ring (with a logged warning) rather than
    letting an unreachable IK call blow up downstream.

    If collect_dropped is True, also returns a list of dropped points,
    each tagged with "stroke_index" set to the position of the
    *surviving output* stroke they were adjacent to (not their original
    input-list index, since whole input strokes can vanish if fewer
    than 2 of their points survive) — this lets a caller line dropped
    points up against trajectory.py's stroke_ranges, which is built
    from the same output-stroke ordering. Points dropped from an input
    stroke that itself doesn't survive are carried forward and attached
    to the next surviving stroke (or the last surviving stroke, if none
    follow).
    """
    strokes_arm = []
    dropped = []
    pending_dropped = []
    total_dropped_count = 0

    for stroke in strokes_px:
        mapped = []
        stroke_dropped_here = []
        for col, row in stroke:
            x, y = img_to_arm(col, row, w, h)
            if kinematics.is_reachable(x, y):
                mapped.append((x, y))
            else:
                total_dropped_count += 1
                logger.warning(
                    "dropping unreachable point: pixel=(%s, %s) -> arm=(%.4f, %.4f)",
                    col, row, x, y,
                )
                if collect_dropped:
                    stroke_dropped_here.append({"pixel": [int(col), int(row)], "arm": [x, y]})

        if len(mapped) >= 2:
            output_idx = len(strokes_arm)
            strokes_arm.append(np.array(mapped, dtype=np.float64))
            if collect_dropped:
                for d in pending_dropped + stroke_dropped_here:
                    d["stroke_index"] = output_idx
                    dropped.append(d)
                pending_dropped = []
        elif collect_dropped:
            pending_dropped.extend(stroke_dropped_here)

    if collect_dropped and pending_dropped:
        fallback_idx = len(strokes_arm) - 1 if strokes_arm else 0
        for d in pending_dropped:
            d["stroke_index"] = fallback_idx
            dropped.append(d)

    if total_dropped_count:
        logger.warning("dropped %d unreachable point(s) total", total_dropped_count)

    if collect_dropped:
        return strokes_arm, dropped
    return strokes_arm
