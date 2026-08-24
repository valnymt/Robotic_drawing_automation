"""End-to-end: photo -> line art -> arm-space strokes -> trajectory ->
PyBullet drawing simulation -> MP4.

Usage:
    python 02_simulate_draw.py photo.jpg [--out drawing.mp4] [--gui]
                                [--frames N] [--fps 30]
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
from config import SIM_TIMESTEP

lineart = importlib.import_module("01_make_lineart")

URDF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "draw_arm.urdf")
PEN_TIP_LINK_INDEX = 2  # base_link(-1) -> link1(0) -> link2(1) -> pen_tip(2)
SHOULDER_JOINT = 0
ELBOW_JOINT = 1

CANVAS_Z = 0.0
INK_COLOR = (0, 0, 0)
CANVAS_COLOR = (1, 1, 1)

CAMERA_TARGET = (0.45, 0.0, 0.0)
CAMERA_DISTANCE = 1.0
CAMERA_YAW = 0
CAMERA_PITCH = -60
CAMERA_UP_AXIS = 2
RENDER_WIDTH = 480
RENDER_HEIGHT = 480
FOV_DEG = 60


def build_scene(gui: bool):
    mode = p.GUI if gui else p.DIRECT
    client = p.connect(mode)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    p.loadURDF("plane.urdf")
    arm_id = p.loadURDF(URDF_PATH, basePosition=[0, 0, 0], useFixedBase=True)
    return client, arm_id


def pen_tip_position(arm_id: int) -> tuple[float, float, float]:
    state = p.getLinkState(arm_id, PEN_TIP_LINK_INDEX)
    return state[0]  # world position (x, y, z)


def set_joint_targets(arm_id: int, t1: float, t2: float):
    # Kinematic simulation: angles are set directly each frame (section 6
    # of robotics_theory_reference.md), not driven through a physics PD
    # controller, which would need many steps per waypoint to converge.
    p.resetJointState(arm_id, SHOULDER_JOINT, t1)
    p.resetJointState(arm_id, ELBOW_JOINT, t2)


INK_HALF_THICKNESS = 0.0015
INK_HALF_HEIGHT = 0.0008
MIN_SEGMENT_LENGTH = 1e-5


def add_segment(p1: tuple, p2: tuple, color=INK_COLOR, width: float = 3.0):
    """Draw 'ink' between two 3D points as real, camera-visible geometry.

    NOTE: p.addUserDebugLine() is a visualizer-only overlay and does NOT
    appear in p.getCameraImage() output under a DIRECT (headless)
    connection (confirmed empirically: 0 rendered pixels regardless of
    renderer backend). A thin static box multibody is used instead,
    since it's real scene geometry the camera actually sees.
    """
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    diff = p2 - p1
    length = float(np.linalg.norm(diff))
    if length < MIN_SEGMENT_LENGTH:
        return

    mid = ((p1 + p2) / 2).tolist()
    mid[2] = CANVAS_Z + INK_HALF_HEIGHT
    yaw = float(np.arctan2(diff[1], diff[0]))
    orientation = p.getQuaternionFromEuler([0, 0, yaw])

    half_extents = [length / 2, INK_HALF_THICKNESS, INK_HALF_HEIGHT]
    rgba = list(color) + [1.0] if len(color) == 3 else list(color)
    visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=rgba)
    p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=visual_shape,
        basePosition=mid,
        baseOrientation=orientation,
    )


def capture_frame(gui: bool = False) -> np.ndarray:
    """Render the current scene with a pinhole camera (section 10) and
    return a BGR uint8 frame for cv2.VideoWriter.

    NOTE: ER_BULLET_HARDWARE_OPENGL returns a blank image under a DIRECT
    (headless) connection on this machine (confirmed empirically — 0
    rendered pixels). ER_TINY_RENDERER (software, CPU) works reliably
    headless; hardware OpenGL is only used when an actual GUI window
    (and its GL context) exists.
    """
    view_matrix = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=CAMERA_TARGET,
        distance=CAMERA_DISTANCE,
        yaw=CAMERA_YAW,
        pitch=CAMERA_PITCH,
        roll=0,
        upAxisIndex=CAMERA_UP_AXIS,
    )
    proj_matrix = p.computeProjectionMatrixFOV(
        fov=FOV_DEG,
        aspect=RENDER_WIDTH / RENDER_HEIGHT,
        nearVal=0.05,
        farVal=3.0,
    )
    renderer = p.ER_BULLET_HARDWARE_OPENGL if gui else p.ER_TINY_RENDERER
    _, _, rgba, _, _ = p.getCameraImage(
        RENDER_WIDTH, RENDER_HEIGHT, view_matrix, proj_matrix,
        renderer=renderer,
    )
    rgb = np.reshape(rgba, (RENDER_HEIGHT, RENDER_WIDTH, 4))[:, :, :3].astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def run_pipeline(image_path: str, out_path: str, gui: bool = False,
                  max_frames: int | None = None, fps: int = 30):
    ordered_strokes_px, _, w, h = lineart.process_image(image_path)
    strokes_arm = mapping.strokes_to_arm(ordered_strokes_px, w, h)
    if not strokes_arm:
        raise RuntimeError("no reachable strokes produced from image; nothing to draw")

    samples = trajectory.generate_trajectory(strokes_arm)
    if max_frames is not None:
        samples = samples[:max_frames]

    client, arm_id = build_scene(gui)
    writer = cv2.VideoWriter(
        out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (RENDER_WIDTH, RENDER_HEIGHT)
    )

    try:
        prev_pos = None
        for sample in samples:
            set_joint_targets(arm_id, sample.t1, sample.t2)
            p.stepSimulation()

            cur_pos = pen_tip_position(arm_id)
            if sample.pen_down and prev_pos is not None:
                add_segment(prev_pos, cur_pos)
            prev_pos = cur_pos

            writer.write(capture_frame(gui=gui))
    finally:
        writer.release()
        p.disconnect(client)

    return out_path


def main():
    parser = argparse.ArgumentParser(description="Photo -> simulated drawing video")
    parser.add_argument("image", help="path to input photo")
    parser.add_argument("--out", default=None, help="output MP4 path (default: <image>_drawing.mp4)")
    parser.add_argument("--gui", action="store_true", help="show PyBullet GUI instead of headless")
    parser.add_argument("--frames", type=int, default=None, help="cap the number of trajectory samples (frames)")
    parser.add_argument("--fps", type=int, default=30, help="output video frame rate")
    args = parser.parse_args()

    base, _ = os.path.splitext(args.image)
    out_path = args.out or f"{base}_drawing.mp4"

    run_pipeline(args.image, out_path, gui=args.gui, max_frames=args.frames, fps=args.fps)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
