"""Single entry point for the draw-arm pipeline.

    python run.py photo.jpg --mode sim|verify|live

Chains phases 01 (line art) -> 03 (mapping) -> 04 (trajectory) into
whichever output mode is chosen, with real validation at each boundary:
  - rejects unreadable/non-image files up front, before any processing
  - warns (doesn't crash) if Canny finds zero edges
  - warns and skips individual unreachable points/strokes rather than
    aborting the whole run (handled by mapping.strokes_to_arm)
  - gives one clear, non-traceback error if every stroke ends up
    unreachable, so there's nothing left to draw
"""

import argparse
import importlib
import logging
import os
import sys

import cv2

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger("run")


class PipelineError(Exception):
    """A clean, user-facing pipeline failure (no traceback needed)."""


def validate_image(path: str) -> None:
    if not os.path.isfile(path):
        raise PipelineError(f"no such file: {path}")
    image = cv2.imread(path)
    if image is None:
        raise PipelineError(
            f"could not read '{path}' as an image (unsupported format or corrupt file)"
        )


def check_lineart_nonempty(image_path: str):
    """Runs phase 01 and warns (but does not crash) if Canny found no
    edges at all. Returns the ordered strokes for the caller to reuse."""
    lineart = importlib.import_module("01_make_lineart")
    ordered_strokes_px, preview, w, h = lineart.process_image(image_path)
    if not ordered_strokes_px:
        logger.warning(
            "Canny found zero edges in '%s' — check CANNY_LOW/CANNY_HIGH in "
            "config.py, or that the image has visible contrast/edges.",
            image_path,
        )
    return ordered_strokes_px, preview, w, h


def run_sim(image_path: str, args):
    simulate = importlib.import_module("02_simulate_draw")
    check_lineart_nonempty(image_path)  # early warning before the sim spins up
    out_path = args.out or f"{os.path.splitext(image_path)[0]}_drawing.mp4"
    try:
        simulate.run_pipeline(image_path, out_path, gui=args.gui, max_frames=args.frames, fps=args.fps)
    except RuntimeError as e:
        raise PipelineError(str(e)) from e
    print(f"wrote {out_path}")
    return out_path


def run_verify(image_path: str, args):
    verify = importlib.import_module("03_verify_topdown")
    check_lineart_nonempty(image_path)
    try:
        final_path, overlay_path = verify.run_topdown_verify(image_path, args.out_dir)
    except RuntimeError as e:
        raise PipelineError(str(e)) from e
    print(f"wrote {final_path}")
    print(f"wrote {overlay_path}")
    return final_path, overlay_path


def run_live(image_path: str, args):
    dashboard = importlib.import_module("04_live_dashboard")
    check_lineart_nonempty(image_path)
    try:
        dashboard.load_pipeline(image_path)
    except RuntimeError as e:
        raise PipelineError(str(e)) from e

    import threading
    thread = threading.Thread(target=dashboard.playback_worker, args=(args.speed,), daemon=True)
    thread.start()

    import uvicorn
    print(f"serving live dashboard at http://127.0.0.1:{args.port}/")
    uvicorn.run(dashboard.app, host="127.0.0.1", port=args.port)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="draw-arm pipeline entry point")
    parser.add_argument("image", help="path to input photo")
    parser.add_argument("--mode", choices=["sim", "verify", "live"], default="sim")

    # sim-specific
    parser.add_argument("--out", default=None, help="[sim] output MP4 path")
    parser.add_argument("--gui", action="store_true", help="[sim] show PyBullet GUI")
    parser.add_argument("--frames", type=int, default=None, help="[sim] cap trajectory samples")
    parser.add_argument("--fps", type=int, default=30, help="[sim] output video frame rate")

    # verify-specific
    parser.add_argument("--out-dir", default=".", help="[verify] directory to write outputs to")

    # live-specific
    parser.add_argument("--port", type=int, default=8000, help="[live] server port")
    parser.add_argument("--speed", type=float, default=1.0, help="[live] playback speed multiplier")

    return parser


def main():
    args = build_parser().parse_args()

    try:
        validate_image(args.image)

        if args.mode == "sim":
            run_sim(args.image, args)
        elif args.mode == "verify":
            run_verify(args.image, args)
        elif args.mode == "live":
            run_live(args.image, args)
    except PipelineError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
