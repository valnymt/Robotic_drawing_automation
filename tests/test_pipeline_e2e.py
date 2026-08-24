import os
import subprocess
import sys
import time
import urllib.request

import cv2
import numpy as np
import pytest

pytest.importorskip("pybullet")

import importlib
run = importlib.import_module("run")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def make_sample_image(path, size=200):
    img = np.full((size, size, 3), 255, dtype=np.uint8)
    cv2.circle(img, (size // 2, size // 2), size // 3, (0, 0, 0), thickness=2)
    cv2.rectangle(img, (40, 40), (160, 160), (0, 0, 0), thickness=2)
    cv2.imwrite(path, img)


class Args:
    """Minimal stand-in for argparse.Namespace with defaults run.py expects."""
    def __init__(self, **overrides):
        self.out = None
        self.gui = False
        self.frames = 15
        self.fps = 10
        self.out_dir = "."
        self.port = 8765
        self.speed = 1.0
        self.__dict__.update(overrides)


def test_validate_image_rejects_missing_file():
    with pytest.raises(run.PipelineError):
        run.validate_image("does_not_exist.png")


def test_validate_image_rejects_non_image_file(tmp_path):
    bad = tmp_path / "not_an_image.txt"
    bad.write_text("this is not an image")
    with pytest.raises(run.PipelineError):
        run.validate_image(str(bad))


def test_validate_image_accepts_real_image(tmp_path):
    img_path = str(tmp_path / "ok.png")
    make_sample_image(img_path)
    run.validate_image(img_path)  # should not raise


def test_run_sim_produces_video(tmp_path):
    img_path = str(tmp_path / "sample.png")
    make_sample_image(img_path)
    out_path = str(tmp_path / "out.mp4")
    args = Args(out=out_path)

    result = run.run_sim(img_path, args)

    assert result == out_path
    assert os.path.exists(out_path)
    assert os.path.getsize(out_path) > 0


def test_run_verify_produces_images(tmp_path):
    img_path = str(tmp_path / "sample.png")
    make_sample_image(img_path)
    args = Args(out_dir=str(tmp_path))

    final_path, overlay_path = run.run_verify(img_path, args)

    assert os.path.exists(final_path)
    assert os.path.getsize(final_path) > 0
    assert os.path.exists(overlay_path)
    assert os.path.getsize(overlay_path) > 0


def test_run_live_serves_dashboard(tmp_path):
    img_path = str(tmp_path / "sample.png")
    make_sample_image(img_path)
    port = 8766

    proc = subprocess.Popen(
        [sys.executable, os.path.join(REPO_ROOT, "run.py"), img_path,
         "--mode", "live", "--port", str(port)],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    try:
        deadline = time.time() + 30
        last_error = None
        responded = False
        while time.time() < deadline:
            if proc.poll() is not None:
                # Process died before ever serving; surface its output.
                output = proc.stdout.read()
                pytest.fail(f"run.py --mode live exited early:\n{output}")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/meta", timeout=1) as resp:
                    assert resp.status == 200
                    responded = True
                    break
            except Exception as e:
                last_error = e
                time.sleep(0.5)
        assert responded, f"server never responded, last error: {last_error}"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def test_run_sim_gives_clean_error_when_nothing_reachable(tmp_path, monkeypatch):
    """Every stroke ending up unreachable should raise a clean
    PipelineError, not an unhandled exception."""
    img_path = str(tmp_path / "sample.png")
    make_sample_image(img_path)

    mapping = importlib.import_module("mapping")
    monkeypatch.setattr(mapping.kinematics, "is_reachable", lambda x, y: False)

    args = Args(out=str(tmp_path / "out.mp4"))
    with pytest.raises(run.PipelineError):
        run.run_sim(img_path, args)
