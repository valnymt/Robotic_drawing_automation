import numpy as np
import pytest

import kinematics as kin
from config import L1, L2, R_MIN, R_MAX


def reachable_grid():
    """Grid of (x, y) points strictly inside the reachable ring."""
    points = []
    for r in np.linspace(R_MIN + 0.02, R_MAX - 0.02, 6):
        for theta in np.linspace(-np.pi + 0.1, np.pi - 0.1, 8):
            points.append((r * np.cos(theta), r * np.sin(theta)))
    return points


def test_fk_ik_round_trip():
    for x, y in reachable_grid():
        t1, t2 = kin.ik(x, y)
        x2, y2, _ = kin.fk(t1, t2)
        assert x2 == pytest.approx(x, abs=1e-6)
        assert y2 == pytest.approx(y, abs=1e-6)


def test_fk_matches_known_pose():
    # theta1=0, theta2=0 -> fully stretched along +x
    x, y, phi = kin.fk(0.0, 0.0)
    assert x == pytest.approx(L1 + L2, abs=1e-9)
    assert y == pytest.approx(0.0, abs=1e-9)
    assert phi == pytest.approx(0.0, abs=1e-9)


def test_ik_elbow_down():
    # theta2 should always be <= 0 for the elbow-down solution
    for x, y in reachable_grid():
        _, t2 = kin.ik(x, y)
        assert t2 <= 1e-9


@pytest.mark.parametrize(
    "x,y",
    [
        (0.0, 0.0),          # inside R_MIN
        (0.05, 0.0),         # inside R_MIN
        (L1 + L2 + 0.5, 0.0),  # beyond R_MAX
        (0.0, 10.0),          # way beyond R_MAX
    ],
)
def test_ik_raises_for_unreachable(x, y):
    assert not kin.is_reachable(x, y)
    with pytest.raises(kin.UnreachableError):
        kin.ik(x, y)


def test_is_reachable_boundary():
    assert kin.is_reachable(R_MIN, 0.0)
    assert kin.is_reachable(R_MAX, 0.0)
    assert not kin.is_reachable(R_MIN - 0.01, 0.0)
    assert not kin.is_reachable(R_MAX + 0.01, 0.0)


def test_is_singular_stretched_and_folded():
    assert kin.is_singular(0.3, 0.0)          # theta2 = 0, fully stretched
    assert kin.is_singular(0.3, np.pi)        # theta2 = pi, fully folded
    assert kin.is_singular(0.3, -np.pi)
    assert not kin.is_singular(0.3, -1.0)     # ordinary configuration


def test_ik_matches_singularity_at_ring_boundary():
    # r = R_MAX (fully stretched) should land ik on theta2 = 0
    t1, t2 = kin.ik(R_MAX, 0.0)
    assert kin.is_singular(t1, t2)
    # r = R_MIN (fully folded) should land ik on theta2 = +-pi
    t1, t2 = kin.ik(R_MIN, 0.0)
    assert kin.is_singular(t1, t2)
