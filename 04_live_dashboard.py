"""Live dashboard: upload any photo, watch the arm decide and draw it
in real time.

Usage:
    python 04_live_dashboard.py [--port 8000] [--speed 1.0] [photo.jpg]

Then open http://127.0.0.1:8000/ and drop a photo onto the page — or
pass one on the command line to start already loaded. Uploading a new
photo at any time resets the run.
"""

import argparse
import asyncio
import base64
import importlib
import json
import os
import queue
import shutil
import tempfile
import threading
import time

import cv2
import numpy as np
import pybullet as p
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import kinematics
import mapping
import trajectory
from config import (
    L1, L2, X0, X1, Y0, Y1, IMG_TARGET_SIZE,
    DRAW_SECONDS_PER_SAMPLE, TRAVEL_SECONDS_PER_SAMPLE,
    CANNY_LOW, CANNY_HIGH, GAUSSIAN_BLUR_KSIZE,
)

lineart = importlib.import_module("01_make_lineart")
simulate = importlib.import_module("02_simulate_draw")

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
UPLOAD_DIR = os.path.join(tempfile.gettempdir(), "drawarm_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# A travel run (consecutive pen-up samples) of at least this many
# samples gets its middle samples collapsed — broadcast only the run's
# first and last sample so the arm visually "jumps" across travel
# instead of crawling through every interpolated waypoint.
MIN_TRAVEL_RUN_TO_COLLAPSE = 3

# Resolution of the server-side ink/target masks used for the live
# "% traced" accuracy metric — matches dashboard.html's inkCanvas
# dimensions exactly, since the accuracy comparison must happen in the
# same pixel space the client draws ink into.
RENDER_WIDTH = 400
RENDER_HEIGHT = 400
METRIC_EVERY_N_SAMPLES = 5
INK_LINE_THICKNESS = 2
TARGET_MASK_DILATE_KSIZE = 5  # ~2px growth per side, so near-miss ink still counts

# --- Real simulated 3D arm frames ---
# A single long-lived PyBullet DIRECT connection, built once at server
# startup and reused across every upload/generation (rebuilding the
# scene per-run is not free). All p.* calls anywhere in this file must
# go through sim_lock, since DIRECT mode has no built-in thread safety
# of its own for concurrent access.
sim_lock = threading.Lock()
SIM_CLIENT = None
SIM_ARM_ID = None
SIM_INK_BODY_IDS: list[int] = []  # tracked so a new upload can clear old ink
FRAME3D_EVERY_N_SAMPLES = 4

app = FastAPI()
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# --- Shared pipeline state. Reset wholesale on every new upload. ---
state = {
    "generation": 0,       # bumped on every new image; stale threads self-cancel
    "image_bytes": None,
    "strokes_px": None,    # list of Nx2 pixel-space polylines (for panel 2)
    "stroke_ranges": None, # per-sample -> stroke index
    "samples": None,
    "history": [],         # sample dicts already broadcast (for reconnects)
    "done": False,
    "loaded": False,
    "w": None, "h": None,
    "default_speed": 1.0,
    "vision_stages": None,  # dict of stage-name -> base64 PNG, for the upload reveal
    "dropped_by_stroke": {},  # output stroke_index -> list of dropped-point dicts
    "target_mask": None,     # bool array, RENDER_HEIGHT x RENDER_WIDTH: where the line art actually is
    "ink_mask": None,        # bool array, same shape: where ink has actually been drawn so far
    "ink_buffer_u8": None,   # uint8 working buffer cv2.line draws into; ink_mask is derived from it
    "last_traced_pct": 0.0,
    # Pause/step control. playback_worker() (the single producer thread)
    # and the FastAPI request handlers (consumers, on the asyncio event
    # loop thread) read/write these directly with no extra lock — safe
    # here because they're simple bool/int flags and CPython's GIL
    # makes each individual read/write atomic; nothing more elaborate
    # than "set this bool" or "increment this int" ever happens to them.
    "paused": False,
    "step_tokens": 0,
}


def arm_to_ink_px(x: float, y: float) -> tuple[int, int]:
    """Arm-space (x, y) -> ink-canvas pixel coords. Must exactly match
    dashboard.html's armToInkPx() JS function, since the accuracy
    metric compares server-drawn ink against the same canvas the
    client renders — including the y-flip (arm-space up = canvas up)."""
    fx = (x - X0) / (X1 - X0)
    fy = (y - Y0) / (Y1 - Y0)
    return int(round(fx * RENDER_WIDTH)), int(round((1 - fy) * RENDER_HEIGHT))
state_lock = threading.Lock()

connected: set[WebSocket] = set()
broadcast_queue: "queue.Queue[dict]" = queue.Queue()


def load_pipeline(image_path: str) -> int:
    """Runs phases 01->03->04 on image_path, replaces the shared state,
    and returns the new generation number (used to invalidate any
    in-flight playback thread from a previous image)."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"could not read image: {image_path}")

    resized = cv2.resize(image, IMG_TARGET_SIZE)
    stages_result = lineart.extract_strokes_with_stages(resized)
    ordered_strokes_px = stages_result["ordered_strokes"]
    w, h = stages_result["width"], stages_result["height"]

    vision_stages_b64 = {
        name: base64.b64encode(cv2.imencode(".png", img)[1].tobytes()).decode("ascii")
        for name, img in stages_result["stages"].items()
    }

    strokes_arm, dropped = mapping.strokes_to_arm(ordered_strokes_px, w, h, collect_dropped=True)
    if not strokes_arm:
        raise RuntimeError("no reachable strokes produced from image; nothing to draw")

    samples = trajectory.generate_trajectory(strokes_arm)
    stroke_index_per_sample = _assign_stroke_indices(strokes_arm, samples)

    dropped_by_stroke: dict[int, list[dict]] = {}
    for d in dropped:
        dropped_by_stroke.setdefault(d["stroke_index"], []).append(d)

    # Target mask: what the finished drawing should look like, at ink-canvas
    # resolution, so live-progress comparisons happen in the same pixel
    # space the ink is actually drawn into.
    gray_for_mask = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gray_for_mask = cv2.resize(gray_for_mask, (RENDER_WIDTH, RENDER_HEIGHT))
    blurred_for_mask = cv2.GaussianBlur(gray_for_mask, GAUSSIAN_BLUR_KSIZE, 0)
    edges_for_mask = cv2.Canny(blurred_for_mask, CANNY_LOW, CANNY_HIGH)
    dilate_kernel = np.ones((TARGET_MASK_DILATE_KSIZE, TARGET_MASK_DILATE_KSIZE), np.uint8)
    target_mask = cv2.dilate(edges_for_mask, dilate_kernel, iterations=1) > 0
    ink_buffer_u8 = np.zeros((RENDER_HEIGHT, RENDER_WIDTH), dtype=np.uint8)

    with state_lock:
        state["generation"] += 1
        gen = state["generation"]
        state["image_bytes"] = cv2.imencode(".png", image)[1].tobytes()
        state["vision_stages"] = vision_stages_b64
        state["strokes_px"] = [s.tolist() for s in ordered_strokes_px]
        state["samples"] = samples
        state["stroke_ranges"] = stroke_index_per_sample
        state["dropped_by_stroke"] = dropped_by_stroke
        state["target_mask"] = target_mask
        state["ink_buffer_u8"] = ink_buffer_u8
        state["ink_mask"] = ink_buffer_u8 > 0
        state["last_traced_pct"] = 0.0
        state["w"], state["h"] = w, h
        state["history"] = []
        state["done"] = False
        state["loaded"] = True
        state["paused"] = False
        state["step_tokens"] = 0

    # Reset the 3D sim's ink so a new drawing doesn't overlay onto a
    # stale one — the scene/arm itself is long-lived and reused, only
    # the drawn ink geometry gets cleared.
    with sim_lock:
        for body_id in SIM_INK_BODY_IDS:
            p.removeBody(body_id)
        SIM_INK_BODY_IDS.clear()

    return gen


def _assign_stroke_indices(strokes_arm, samples):
    """Best-effort mapping of sample index -> stroke index, for UI
    highlighting only (not used for any drawing math)."""
    indices = []
    stroke_idx = 0
    pen_down_seen_in_stroke = False
    for s in samples:
        if s.pen_down:
            pen_down_seen_in_stroke = True
            indices.append(stroke_idx)
        else:
            if pen_down_seen_in_stroke:
                stroke_idx = min(stroke_idx + 1, len(strokes_arm) - 1)
                pen_down_seen_in_stroke = False
            indices.append(stroke_idx)
    return indices


def _sample_to_dict(i: int) -> dict:
    s = state["samples"][i]
    x, y, _ = kinematics.fk(s.t1, s.t2)
    return {
        "type": "sample",
        "frame": i,
        "total": len(state["samples"]),
        "stroke_index": state["stroke_ranges"][i],
        "t1": s.t1,
        "t2": s.t2,
        "x": x,
        "y": y,
        "pen_down": s.pen_down,
    }


def _compute_collapse_flags(samples) -> list[bool]:
    """Marks the *middle* samples of long pen-up travel runs (length >=
    MIN_TRAVEL_RUN_TO_COLLAPSE) as True. The first and last sample of
    such a run are left False (still broadcast normally) so the arm
    visibly jumps from the run's start to its end instead of crawling
    through every interpolated travel waypoint. Runs of pen-down
    samples, and short travel runs, are never collapsed."""
    n = len(samples)
    collapse = [False] * n
    i = 0
    while i < n:
        if not samples[i].pen_down:
            j = i
            while j < n and not samples[j].pen_down:
                j += 1
            run_len = j - i
            if run_len >= MIN_TRAVEL_RUN_TO_COLLAPSE:
                for k in range(i + 1, j - 1):
                    collapse[k] = True
            i = j
        else:
            i += 1
    return collapse


def playback_worker(speed: float, generation: int):
    """Runs in a background thread: walks the trajectory at a watchable
    pace and pushes each sample onto broadcast_queue. Self-cancels if a
    newer image gets loaded mid-run (generation mismatch).

    Pacing is non-uniform: pen-down (drawing) samples play slow and
    deliberate; pen-up (travel) samples play fast, and long travel runs
    have their middle samples collapsed entirely — broadcast (and
    wait for) only the run's first and last sample, so the arm jumps
    across empty transit instead of crawling through it. Every sample
    is still appended to state["history"] regardless, so a
    reconnecting client's replay stays frame-accurate."""
    samples = state["samples"]
    stroke_ranges = state["stroke_ranges"]
    dropped_by_stroke = state["dropped_by_stroke"]
    target_mask = state["target_mask"]
    ink_buffer = state["ink_buffer_u8"]
    collapse = _compute_collapse_flags(samples)
    total = len(samples)
    target_pixel_count = max(int(np.count_nonzero(target_mask)), 1)
    prev_stroke_index = None
    prev_arm_point = None
    prev_pos_3d = None

    for i in range(total):
        if state["generation"] != generation:
            return  # a newer photo was uploaded; this run is stale

        # Pause/step gate: not paused -> proceed immediately; paused with
        # no step tokens -> block here; paused with a step token
        # available -> consume exactly one token and emit this one
        # sample, then the next loop iteration blocks again.
        while state["paused"] and state["step_tokens"] <= 0:
            if state["generation"] != generation:
                return  # a newer photo was uploaded while paused
            time.sleep(0.05)
        if state["step_tokens"] > 0:
            state["step_tokens"] -= 1

        stroke_index = stroke_ranges[i]
        is_stroke_boundary = stroke_index != prev_stroke_index
        if is_stroke_boundary:
            for d in dropped_by_stroke.get(stroke_index, []):
                x, y = d["arm"]
                warn_msg = {
                    "type": "warning",
                    "text": f"⚠ point ({x:.3f}, {y:.3f}) unreachable — skipped",
                    "stroke_index": stroke_index,
                }
                state["history"].append(warn_msg)
                broadcast_queue.put(warn_msg)
            prev_stroke_index = stroke_index

        msg = _sample_to_dict(i)
        state["history"].append(msg)

        # Mirror the client's ink drawing server-side, in the same
        # ink-canvas pixel space, so the accuracy metric below is
        # comparing like-for-like against target_mask.
        sample = samples[i]
        if sample.pen_down and prev_arm_point is not None:
            p0 = arm_to_ink_px(*prev_arm_point)
            p1 = arm_to_ink_px(msg["x"], msg["y"])
            cv2.line(ink_buffer, p0, p1, 255, INK_LINE_THICKNESS)
            state["ink_mask"] = ink_buffer > 0
        prev_arm_point = (msg["x"], msg["y"])

        # Drive the real PyBullet-simulated arm the same way
        # 02_simulate_draw.py's own loop does, and periodically stream a
        # rendered frame — this is what the dashboard's "3D sim" panel
        # shows, not just a flat trig sketch.
        frame3d_b64 = None
        if SIM_ARM_ID is not None:
            with sim_lock:
                simulate.set_joint_targets(SIM_ARM_ID, sample.t1, sample.t2)
                p.stepSimulation()
                cur_pos_3d = simulate.pen_tip_position(SIM_ARM_ID)
                if sample.pen_down and prev_pos_3d is not None:
                    body_id = simulate.add_segment(prev_pos_3d, cur_pos_3d)
                    if body_id is not None:
                        SIM_INK_BODY_IDS.append(body_id)
                prev_pos_3d = cur_pos_3d

                if i % FRAME3D_EVERY_N_SAMPLES == 0:
                    frame = simulate.capture_frame(gui=False)
                    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    if ok:
                        frame3d_b64 = base64.b64encode(buf.tobytes()).decode("ascii")
            if frame3d_b64 is not None:
                broadcast_queue.put({"type": "frame3d", "image_b64": frame3d_b64, "frame": i})

        # Metric computation is an O(w*h) numpy op — throttle it to every
        # few samples (or a stroke boundary) rather than every sample.
        if i % METRIC_EVERY_N_SAMPLES == 0 or is_stroke_boundary:
            traced_pct = 100.0 * np.count_nonzero(state["ink_mask"] & target_mask) / target_pixel_count
            state["last_traced_pct"] = traced_pct
            metric_msg = {"type": "metric", "traced_pct": traced_pct, "frame": i}
            state["history"].append(metric_msg)
            broadcast_queue.put(metric_msg)

        if collapse[i]:
            continue  # collapsed travel-run middle: recorded, not broadcast, no delay

        broadcast_queue.put(msg)
        base_delay = DRAW_SECONDS_PER_SAMPLE if sample.pen_down else TRAVEL_SECONDS_PER_SAMPLE
        time.sleep(base_delay / max(speed, 1e-6))

    if state["generation"] == generation:
        state["done"] = True
        broadcast_queue.put({"type": "done", "traced_pct": state["last_traced_pct"]})


def start_playback(speed: float):
    gen = state["generation"]
    thread = threading.Thread(target=playback_worker, args=(speed, gen), daemon=True)
    thread.start()


async def broadcaster():
    """Pulls from the thread-safe queue and fans out to all connected
    WebSocket clients."""
    while True:
        try:
            msg = broadcast_queue.get_nowait()
        except queue.Empty:
            await asyncio.sleep(0.005)
            continue
        dead = []
        for ws in connected:
            try:
                await ws.send_text(json.dumps(msg))
            except Exception:
                dead.append(ws)
        for ws in dead:
            connected.discard(ws)


@app.on_event("startup")
async def on_startup():
    global SIM_CLIENT, SIM_ARM_ID
    asyncio.create_task(broadcaster())
    with sim_lock:
        SIM_CLIENT, SIM_ARM_ID = simulate.build_scene(gui=False)


@app.get("/")
async def index():
    return FileResponse(os.path.join(STATIC_DIR, "dashboard.html"))


@app.get("/api/meta")
async def api_meta():
    with state_lock:
        return JSONResponse({
            "loaded": state["loaded"],
            "L1": L1, "L2": L2,
            "X0": X0, "X1": X1, "Y0": Y0, "Y1": Y1,
            "img_w": state["w"], "img_h": state["h"],
            "total_samples": len(state["samples"]) if state["samples"] else 0,
            "generation": state["generation"],
        })


@app.get("/api/photo")
async def api_photo():
    if not state["loaded"]:
        return JSONResponse({"image_b64": None})
    return JSONResponse({"image_b64": base64.b64encode(state["image_bytes"]).decode("ascii")})


@app.get("/api/strokes")
async def api_strokes():
    return JSONResponse({"strokes": state["strokes_px"] or []})


@app.get("/api/vision-stages")
async def api_vision_stages():
    return JSONResponse({"stages": state["vision_stages"] or {}})


@app.post("/api/upload")
async def api_upload(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename or "")[1] or ".png"
    dest_path = os.path.join(UPLOAD_DIR, f"upload_{int(time.time() * 1000)}{ext}")
    with open(dest_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        gen = load_pipeline(dest_path)
    except FileNotFoundError:
        return JSONResponse({"error": "could not read that file as an image"}, status_code=400)
    except RuntimeError as e:
        return JSONResponse({"error": str(e)}, status_code=422)

    # Tell every connected client to wipe their canvases now, but do NOT
    # start playback yet — the frontend plays the vision-stages reveal
    # first and calls /api/start-playback once that finishes.
    broadcast_queue.put({"type": "reset", "generation": gen})

    return JSONResponse({"ok": True, "generation": gen, "total_samples": len(state["samples"])})


@app.post("/api/start-playback")
async def api_start_playback(speed: float = 1.0):
    if not state["loaded"]:
        return JSONResponse({"error": "no image loaded yet"}, status_code=400)
    start_playback(speed)
    return JSONResponse({"ok": True, "generation": state["generation"]})


@app.post("/api/control/pause")
async def api_control_pause():
    if not state["loaded"]:
        return JSONResponse({"error": "no image loaded yet"}, status_code=400)
    state["paused"] = True
    broadcast_queue.put({"type": "playstate", "paused": True})
    return JSONResponse({"ok": True, "paused": True})


@app.post("/api/control/resume")
async def api_control_resume():
    if not state["loaded"]:
        return JSONResponse({"error": "no image loaded yet"}, status_code=400)
    state["paused"] = False
    broadcast_queue.put({"type": "playstate", "paused": False})
    return JSONResponse({"ok": True, "paused": False})


@app.post("/api/control/step")
async def api_control_step():
    if not state["loaded"]:
        return JSONResponse({"error": "no image loaded yet"}, status_code=400)
    state["step_tokens"] += 1
    return JSONResponse({"ok": True, "step_tokens": state["step_tokens"]})


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected.add(websocket)
    try:
        with state_lock:
            await websocket.send_text(json.dumps({
                "type": "history",
                "loaded": state["loaded"],
                "generation": state["generation"],
                "samples": state["history"],
                "done": state["done"],
                "paused": state["paused"],
            }))
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        connected.discard(websocket)


def main():
    parser = argparse.ArgumentParser(description="Live draw-arm dashboard")
    parser.add_argument("image", nargs="?", default=None, help="optional path to a starting photo")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--speed", type=float, default=1.0, help="playback speed multiplier")
    args = parser.parse_args()

    state["default_speed"] = args.speed

    if args.image:
        load_pipeline(args.image)
        start_playback(args.speed)

    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
