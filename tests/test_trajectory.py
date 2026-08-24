import numpy as np
import pytest

import kinematics
import trajectory
from config import R_MIN, R_MAX

MAX_STEP_JUMP = 0.5  # radians; generous bound, just catches actual discontinuities


def ring_point(r_frac: float, theta: float) -> tuple[float, float]:
    r = R_MIN + r_frac * (R_MAX - R_MIN)
    return r * np.cos(theta), r * np.sin(theta)


def make_strokes():
    stroke_a = np.array([
        ring_point(0.5, 0.2),
        ring_point(0.55, 0.3),
        ring_point(0.6, 0.4),
    ])
    stroke_b = np.array([
        ring_point(0.4, -0.5),
        ring_point(0.45, -0.6),
    ])
    return [stroke_a, stroke_b]


def test_theta_arrays_continuous():
    strokes = make_strokes()
    samples = trajectory.generate_trajectory(strokes, steps_per_segment=4)

    for prev, cur in zip(samples, samples[1:]):
        assert abs(cur.t1 - prev.t1) < MAX_STEP_JUMP
        assert abs(cur.t2 - prev.t2) < MAX_STEP_JUMP


def test_pen_down_flag_pattern():
    strokes = make_strokes()
    samples = trajectory.generate_trajectory(strokes, steps_per_segment=3)

    assert any(s.pen_down for s in samples)
    assert any(not s.pen_down for s in samples)
    # first sample (initial travel to first stroke) must be pen-up
    assert samples[0].pen_down is False


def test_fk_matches_stroke_endpoints():
    strokes = make_strokes()
    samples = trajectory.generate_trajectory(strokes, steps_per_segment=5)

    # Reconstruct: the last pen-down sample before a pen-up run should
    # match the corresponding stroke's final waypoint via fk().
    stroke_ends = [stroke[-1] for stroke in strokes]

    pen_down_runs = []
    current_run = []
    for s in samples:
        if s.pen_down:
            current_run.append(s)
        else:
            if current_run:
                pen_down_runs.append(current_run)
                current_run = []
    if current_run:
        pen_down_runs.append(current_run)

    assert len(pen_down_runs) == len(strokes)

    for run, expected_end in zip(pen_down_runs, stroke_ends):
        last = run[-1]
        x, y, _ = kinematics.fk(last.t1, last.t2)
        assert x == pytest.approx(expected_end[0], abs=1e-6)
        assert y == pytest.approx(expected_end[1], abs=1e-6)


def test_quintic_mode_also_continuous_and_accurate():
    strokes = make_strokes()
    samples = trajectory.generate_trajectory(strokes, steps_per_segment=6, quintic=True)

    for prev, cur in zip(samples, samples[1:]):
        assert abs(cur.t1 - prev.t1) < MAX_STEP_JUMP
        assert abs(cur.t2 - prev.t2) < MAX_STEP_JUMP

    last = samples[-1]
    x, y, _ = kinematics.fk(last.t1, last.t2)
    expected = strokes[-1][-1]
    assert x == pytest.approx(expected[0], abs=1e-6)
    assert y == pytest.approx(expected[1], abs=1e-6)


def test_empty_input_returns_empty():
    assert trajectory.generate_trajectory([]) == []
