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
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import kinematics
import mapping
import trajectory
from config import L1, L2, X0, X1, Y0, Y1, IMG_TARGET_SIZE

lineart = importlib.import_module("01_make_lineart")

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
UPLOAD_DIR = os.path.join(tempfile.gettempdir(), "drawarm_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
SECONDS_PER_SAMPLE = 0.02  # base playback pace; divided by --speed

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
}
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

    with state_lock:
        state["generation"] += 1
        gen = state["generation"]
        state["image_bytes"] = cv2.imencode(".png", image)[1].tobytes()
        state["vision_stages"] = vision_stages_b64
        state["strokes_px"] = [s.tolist() for s in ordered_strokes_px]
        state["samples"] = samples
        state["stroke_ranges"] = stroke_index_per_sample
        state["dropped_by_stroke"] = dropped_by_stroke
        state["w"], state["h"] = w, h
        state["history"] = []
        state["done"] = False
        state["loaded"] = True

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


def playback_worker(speed: float, generation: int):
    """Runs in a background thread: walks the trajectory at a watchable
    pace and pushes each sample onto broadcast_queue. Self-cancels if a
    newer image gets loaded mid-run (generation mismatch)."""
    delay = SECONDS_PER_SAMPLE / max(speed, 1e-6)
    samples = state["samples"]
    stroke_ranges = state["stroke_ranges"]
    dropped_by_stroke = state["dropped_by_stroke"]
    total = len(samples)
    prev_stroke_index = None
    for i in range(total):
        if state["generation"] != generation:
            return  # a newer photo was uploaded; this run is stale

        stroke_index = stroke_ranges[i]
        if stroke_index != prev_stroke_index:
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
        broadcast_queue.put(msg)
        time.sleep(delay)
    if state["generation"] == generation:
        state["done"] = True
        broadcast_queue.put({"type": "done"})


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
    asyncio.create_task(broadcaster())


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
