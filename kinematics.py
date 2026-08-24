"""Pure kinematics for the 2-DOF planar RR arm.

Forward kinematics, closed-form inverse kinematics (elbow-down root),
and reachability/singularity checks. See robotics_theory_reference.md
sections 3-5 for the derivations.
"""

import numpy as np

from config import L1, L2, R_MIN, R_MAX

SINGULARITY_EPS = 1e-6


class UnreachableError(ValueError):
    """Raised by ik() when (x, y) is outside the arm's reachable ring."""


def fk(t1: float, t2: float) -> tuple[float, float, float]:
    """Joint angles -> pen-tip (x, y, phi)."""
    x = L1 * np.cos(t1) + L2 * np.cos(t1 + t2)
    y = L1 * np.sin(t1) + L2 * np.sin(t1 + t2)
    phi = t1 + t2
    return x, y, phi


def is_reachable(x: float, y: float) -> bool:
    r = np.hypot(x, y)
    return R_MIN <= r <= R_MAX


def ik(x: float, y: float) -> tuple[float, float]:
    """Pen-tip (x, y) -> joint angles (t1, t2), elbow-down solution.

    Raises UnreachableError if (x, y) is outside [R_MIN, R_MAX].
    """
    if not is_reachable(x, y):
        raise UnreachableError(
            f"point ({x}, {y}) at r={np.hypot(x, y):.4f} is outside "
            f"the reachable ring [{R_MIN:.4f}, {R_MAX:.4f}]"
        )

    cos_t2 = (x**2 + y**2 - L1**2 - L2**2) / (2 * L1 * L2)
    cos_t2 = np.clip(cos_t2, -1.0, 1.0)
    t2 = -np.arccos(cos_t2)  # elbow-down solution
    t1 = np.arctan2(y, x) - np.arctan2(L2 * np.sin(t2), L1 + L2 * np.cos(t2))
    return float(t1), float(t2)


def is_singular(t1: float, t2: float, eps: float = SINGULARITY_EPS) -> bool:
    """True at theta2 = 0 (fully stretched) or theta2 = +-pi (fully folded),
    where det(J) = L1*L2*sin(theta2) = 0."""
    return abs(np.sin(t2)) < eps
