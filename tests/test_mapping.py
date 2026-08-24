import numpy as np
import pytest

import mapping
from config import X0, X1, Y0, Y1


def test_corner_mapping():
    w, h = 640, 640
    x, y = mapping.img_to_arm(0, 0, w, h)
    assert x == pytest.approx(X0)
    assert y == pytest.approx(Y1)

    x, y = mapping.img_to_arm(w, h, w, h)
    assert x == pytest.approx(X1)
    assert y == pytest.approx(Y0)


def test_center_mapping():
    w, h = 640, 640
    x, y = mapping.img_to_arm(w / 2, h / 2, w, h)
    assert x == pytest.approx((X0 + X1) / 2)
    assert y == pytest.approx((Y0 + Y1) / 2)


def test_strokes_to_arm_keeps_reachable_stroke():
    w, h = 640, 640
    stroke = np.array([[100, 100], [200, 200], [300, 300]])
    result = mapping.strokes_to_arm([stroke], w, h)
    assert len(result) == 1
    assert result[0].shape[1] == 2


def test_strokes_to_arm_drops_unreachable_points(monkeypatch):
    w, h = 640, 640
    # Force is_reachable to reject one specific arm-space point so the
    # drop path is exercised without needing config's bounds to actually
    # reach outside the ring.
    unreachable_col, unreachable_row = 50, 50
    target_x, target_y = mapping.img_to_arm(unreachable_col, unreachable_row, w, h)

    real_is_reachable = mapping.kinematics.is_reachable

    def fake_is_reachable(x, y):
        if np.isclose(x, target_x) and np.isclose(y, target_y):
            return False
        return real_is_reachable(x, y)

    monkeypatch.setattr(mapping.kinematics, "is_reachable", fake_is_reachable)

    stroke = np.array([
        [unreachable_col, unreachable_row],
        [200, 200],
        [300, 300],
    ])
    result = mapping.strokes_to_arm([stroke], w, h)

    assert len(result) == 1
    mapped_points = result[0]
    assert not any(
        np.isclose(p[0], target_x) and np.isclose(p[1], target_y)
        for p in mapped_points
    )
    assert len(mapped_points) == 2


def test_strokes_to_arm_drops_whole_stroke_if_under_two_points(monkeypatch):
    w, h = 640, 640
    monkeypatch.setattr(mapping.kinematics, "is_reachable", lambda x, y: False)
    stroke = np.array([[100, 100], [200, 200]])
    result = mapping.strokes_to_arm([stroke], w, h)
    assert result == []
