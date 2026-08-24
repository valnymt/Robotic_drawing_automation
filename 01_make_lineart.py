"""Photo -> ordered pixel-space strokes.

Pipeline: Gaussian blur -> Canny -> contours -> Douglas-Peucker
simplification -> greedy nearest-neighbor stroke ordering.

Usage:
    python 01_make_lineart.py path/to/photo.jpg [--out out.json] [--preview preview.png]
"""

import argparse
import json
import os

import cv2
import numpy as np

from config import (
    IMG_TARGET_SIZE,
    CANNY_LOW,
    CANNY_HIGH,
    GAUSSIAN_BLUR_KSIZE,
    APPROX_POLY_EPS_FRAC,
)

MIN_CONTOUR_POINTS = 2


def extract_strokes_with_stages(image: np.ndarray) -> dict:
    """Photo (BGR or grayscale) -> every intermediate stage of the CV
    pipeline, so a caller (e.g. the live dashboard's "vision reveal")
    can show how the final stroke plan was actually reached, not just
    the end result.

    Returns:
        {
          "raw_strokes": list of Nx2 int32 polylines, pre-ordering
          "ordered_strokes": same strokes, greedy-nearest-neighbor ordered
          "width": int, "height": int,
          "stages": {
              "blurred":      BGR preview of the Gaussian-blurred grayscale,
              "edges":        BGR preview of the Canny edge map,
              "raw_contours": BGR preview of cv2.findContours output,
              "simplified":   BGR preview after approxPolyDP, pre-ordering,
              "ordered":      the existing numbered/colored render_preview(),
          },
        }
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    h, w = gray.shape[:2]

    blurred = cv2.GaussianBlur(gray, GAUSSIAN_BLUR_KSIZE, 0)
    edges = cv2.Canny(blurred, CANNY_LOW, CANNY_HIGH)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    raw_contours_preview = np.full((h, w, 3), 255, dtype=np.uint8)
    cv2.drawContours(raw_contours_preview, contours, -1, (150, 150, 150), 1)

    raw_strokes = []
    for contour in contours:
        arc_len = cv2.arcLength(contour, closed=False)
        epsilon = APPROX_POLY_EPS_FRAC * max(arc_len, 1e-6)
        simplified = cv2.approxPolyDP(contour, epsilon, closed=False)
        pts = simplified.reshape(-1, 2)
        if len(pts) >= MIN_CONTOUR_POINTS:
            raw_strokes.append(pts.astype(np.int32))

    simplified_preview = np.full((h, w, 3), 255, dtype=np.uint8)
    for stroke in raw_strokes:
        cv2.polylines(simplified_preview, [stroke], isClosed=False, color=(60, 60, 60), thickness=1)

    ordered_strokes = order_strokes(raw_strokes)
    ordered_preview = render_preview((h, w), ordered_strokes)

    return {
        "raw_strokes": raw_strokes,
        "ordered_strokes": ordered_strokes,
        "width": w,
        "height": h,
        "stages": {
            "blurred": cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR),
            "edges": cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR),
            "raw_contours": raw_contours_preview,
            "simplified": simplified_preview,
            "ordered": ordered_preview,
        },
    }


def extract_strokes(image: np.ndarray) -> list[np.ndarray]:
    """Photo (BGR or grayscale) -> list of Nx2 int32 polylines in pixel
    coords (pre-ordering). Thin wrapper around
    extract_strokes_with_stages() for callers that only need the final
    strokes, not the intermediate stage images."""
    return extract_strokes_with_stages(image)["raw_strokes"]


def order_strokes(strokes: list[np.ndarray], start=(0, 0)) -> list[np.ndarray]:
    """Greedy nearest-neighbor ordering. Reverses a stroke if its tail is
    closer to `current` than its head."""
    remaining = list(strokes)
    ordered = []
    current = np.array(start, dtype=np.float64)

    while remaining:
        best_idx = None
        best_dist = None
        best_reversed = False

        for i, stroke in enumerate(remaining):
            head = stroke[0].astype(np.float64)
            tail = stroke[-1].astype(np.float64)
            d_head = np.linalg.norm(head - current)
            d_tail = np.linalg.norm(tail - current)

            if d_head <= d_tail:
                d, reversed_ = d_head, False
            else:
                d, reversed_ = d_tail, True

            if best_dist is None or d < best_dist:
                best_dist = d
                best_idx = i
                best_reversed = reversed_

        stroke = remaining.pop(best_idx)
        if best_reversed:
            stroke = stroke[::-1]
        ordered.append(stroke)
        current = stroke[-1].astype(np.float64)

    return ordered


def total_travel_distance(strokes: list[np.ndarray], start=(0, 0)) -> float:
    """Sum of pen-up jumps between strokes' endpoints (start of one stroke
    to end of the previous), used to compare ordered vs unordered."""
    current = np.array(start, dtype=np.float64)
    total = 0.0
    for stroke in strokes:
        head = stroke[0].astype(np.float64)
        total += np.linalg.norm(head - current)
        current = stroke[-1].astype(np.float64)
    return total


def render_preview(image_shape, strokes: list[np.ndarray]) -> np.ndarray:
    h, w = image_shape[:2]
    preview = np.full((h, w, 3), 255, dtype=np.uint8)

    for i, stroke in enumerate(strokes):
        color = (
            int(40 + (i * 37) % 200),
            int(40 + (i * 91) % 200),
            int(40 + (i * 53) % 200),
        )
        cv2.polylines(preview, [stroke], isClosed=False, color=color, thickness=1)
        if len(stroke) > 0:
            cv2.putText(
                preview,
                str(i),
                tuple(stroke[0]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                color,
                1,
                cv2.LINE_AA,
            )

    return preview


def strokes_to_json(strokes: list[np.ndarray], width: int, height: int) -> dict:
    return {
        "width": width,
        "height": height,
        "strokes": [stroke.tolist() for stroke in strokes],
    }


def process_image(path: str):
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"could not read image: {path}")

    image = cv2.resize(image, IMG_TARGET_SIZE)
    raw_strokes = extract_strokes(image)
    ordered = order_strokes(raw_strokes)
    preview = render_preview(image.shape, ordered)
    h, w = image.shape[:2]
    return ordered, preview, w, h


def main():
    parser = argparse.ArgumentParser(description="Photo -> ordered line-art strokes")
    parser.add_argument("image", help="path to input photo")
    parser.add_argument("--out", default=None, help="output JSON path (default: <image>_strokes.json)")
    parser.add_argument("--preview", default=None, help="output preview PNG path (default: <image>_preview.png)")
    args = parser.parse_args()

    base, _ = os.path.splitext(args.image)
    out_path = args.out or f"{base}_strokes.json"
    preview_path = args.preview or f"{base}_preview.png"

    ordered, preview, w, h = process_image(args.image)

    with open(out_path, "w") as f:
        json.dump(strokes_to_json(ordered, w, h), f)

    cv2.imwrite(preview_path, preview)

    print(f"strokes: {len(ordered)}")
    print(f"wrote {out_path}")
    print(f"wrote {preview_path}")


if __name__ == "__main__":
    main()
