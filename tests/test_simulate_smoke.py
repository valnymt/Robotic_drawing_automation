import importlib
import os

import cv2
import numpy as np
import pytest

pytest.importorskip("pybullet")

simulate = importlib.import_module("02_simulate_draw")


def make_tiny_synthetic_image(path, size=64):
    img = np.full((size, size, 3), 255, dtype=np.uint8)
    cv2.circle(img, (size // 2, size // 2), size // 3, (0, 0, 0), thickness=2)
    cv2.imwrite(path, img)


def test_pipeline_produces_nonempty_video(tmp_path):
    img_path = str(tmp_path / "tiny.png")
    make_tiny_synthetic_image(img_path)

    out_path = str(tmp_path / "drawing.mp4")
    simulate.run_pipeline(img_path, out_path, gui=False, max_frames=10, fps=10)

    assert os.path.exists(out_path)
    assert os.path.getsize(out_path) > 0
