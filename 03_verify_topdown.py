"""Independent top-down verification render.

Runs the same drawing loop as 02_simulate_draw.py but with a fixed,
straight-down camera framed on the drawing workspace, arm geometry
hidden for the final shot, so the rendered ink can be checked by eye
against the input line art. Reuses img_to_arm() (mapping.py) and
add_segment() (02_simulate_draw.py) rather than reimplementing them.

Usage:
    python 03_verify_topdown.py photo.jpg [--out-dir .]
"""

import argparse
import importlib
import os

import cv2
import numpy as np
import pybullet as p
import pybullet_data

import mapping
import trajectory
from config import X0, X1, Y0, Y1

lineart = importlib.import_module("01_make_lineart")
simulate = importlib.import_module("02_simulate_draw")

RENDER_WIDTH = 480
RENDER_HEIGHT = 480

# Straight-down camera framed to fit the whole drawing workspace, plus
# margin. NOTE: PyBullet's computeProjectionMatrix() (true orthographic)
# renders unusably small in this build regardless of near/far/extent
# settings (confirmed empirically — a 0.2m box occupies a single
# pixel). A perspective camera pulled back with a FOV sized to the
# workspace is used instead; distortion at this distance is negligible
# for by-eye geometric verification.
CANVAS_CENTER = ((X0 + X1) / 2, (Y0 + Y1) / 2, 0.0)
CAMERA_HEIGHT = 1.6
WORKSPACE_MARGIN = 0.08
_half_span = max((X1 - X0) / 2, (Y1 - Y0) / 2) + WORKSPACE_MARGIN
CAMERA_FOV_DEG = 2 * np.degrees(np.arctan(_half_span / CAMERA_HEIGHT)) * 1.15  # +15% breathing room


def build_topdown_matrices():
    view_matrix = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=CANVAS_CENTER,
        distance=CAMERA_HEIGHT,
        yaw=0,
        pitch=-89.999,  # straight down (exactly -90 degenerates the view matrix)
        roll=0,
        upAxisIndex=2,  # world is Z-up (matches draw_arm.urdf / gravity axis)
    )
    proj_matrix = p.computeProjectionMatrixFOV(
        fov=CAMERA_FOV_DEG,
        aspect=RENDER_WIDTH / RENDER_HEIGHT,
        nearVal=0.1,
        farVal=3.0,
    )
    return view_matrix, proj_matrix


def hide_arm(arm_id: int):
    """Make the arm's own links invisible so the final shot shows only
    the drawn ink, not the robot pose it ended on."""
    num_joints = p.getNumJoints(arm_id)
    for link_index in range(-1, num_joints):  # -1 = base link
        p.changeVisualShape(arm_id, link_index, rgbaColor=[0, 0, 0, 0])


def capture_topdown_frame(view_matrix, proj_matrix) -> np.ndarray:
    # ER_TINY_RENDERER (software) works reliably headless; see the note
    # in 02_simulate_draw.py's capture_frame().
    _, _, rgba, _, _ = p.getCameraImage(
        RENDER_WIDTH, RENDER_HEIGHT, view_matrix, proj_matrix,
        renderer=p.ER_TINY_RENDERER,
    )
    rgb = np.reshape(rgba, (RENDER_HEIGHT, RENDER_WIDTH, 4))[:, :, :3].astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def run_topdown_verify(image_path: str, out_dir: str = "."):
    ordered_strokes_px, lineart_preview, w, h = lineart.process_image(image_path)
    strokes_arm = mapping.strokes_to_arm(ordered_strokes_px, w, h)
    if not strokes_arm:
        raise RuntimeError("no reachable strokes produced from image; nothing to draw")

    samples = trajectory.generate_trajectory(strokes_arm)

    client, arm_id = simulate.build_scene(gui=False)
    view_matrix, proj_matrix = build_topdown_matrices()

    try:
        prev_pos = None
        for sample in samples:
            simulate.set_joint_targets(arm_id, sample.t1, sample.t2)
            p.stepSimulation()

            cur_pos = simulate.pen_tip_position(arm_id)
            if sample.pen_down and prev_pos is not None:
                simulate.add_segment(prev_pos, cur_pos)
            prev_pos = cur_pos

        hide_arm(arm_id)
        final_render = capture_topdown_frame(view_matrix, proj_matrix)
    finally:
        p.disconnect(client)

    base = os.path.splitext(os.path.basename(image_path))[0]
    final_path = os.path.join(out_dir, f"{base}_topdown.png")
    overlay_path = os.path.join(out_dir, f"{base}_overlay.png")

    cv2.imwrite(final_path, final_render)

    lineart_resized = cv2.resize(lineart_preview, (RENDER_WIDTH, RENDER_HEIGHT))
    overlay = np.hstack([lineart_resized, final_render])
    cv2.putText(overlay, "input line art", (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(overlay, "rendered drawing", (RENDER_WIDTH + 10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.imwrite(overlay_path, overlay)

    return final_path, overlay_path


def main():
    parser = argparse.ArgumentParser(description="Top-down verification render")
    parser.add_argument("image", help="path to input photo")
    parser.add_argument("--out-dir", default=".", help="directory to write outputs to")
    args = parser.parse_args()

    final_path, overlay_path = run_topdown_verify(args.image, args.out_dir)
    print(f"wrote {final_path}")
    print(f"wrote {overlay_path}")


if __name__ == "__main__":
    main()
