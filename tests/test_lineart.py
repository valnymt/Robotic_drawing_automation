import cv2
import numpy as np

import importlib

lineart = importlib.import_module("01_make_lineart")


def make_synthetic_circle_image(size=200):
    img = np.full((size, size, 3), 255, dtype=np.uint8)
    cv2.circle(img, (size // 2, size // 2), size // 3, (0, 0, 0), thickness=2)
    return img


def test_extract_strokes_nonempty():
    img = make_synthetic_circle_image()
    strokes = lineart.extract_strokes(img)
    assert len(strokes) > 0
    for stroke in strokes:
        assert stroke.shape[1] == 2
        assert len(stroke) >= 2


def test_ordering_reduces_travel_distance():
    img = make_synthetic_circle_image()
    raw_strokes = lineart.extract_strokes(img)
    # Need at least two strokes for ordering to matter; if the circle
    # collapses to one contour, synthesize extra scattered strokes.
    if len(raw_strokes) < 2:
        raw_strokes = raw_strokes + [
            np.array([[190, 10], [195, 15]], dtype=np.int32),
            np.array([[5, 190], [10, 195]], dtype=np.int32),
            np.array([[100, 5], [105, 8]], dtype=np.int32),
        ]

    unordered_dist = lineart.total_travel_distance(raw_strokes)
    ordered = lineart.order_strokes(raw_strokes)
    ordered_dist = lineart.total_travel_distance(ordered)

    assert len(ordered) == len(raw_strokes)
    assert ordered_dist <= unordered_dist


def test_process_image_end_to_end(tmp_path):
    img = make_synthetic_circle_image()
    img_path = tmp_path / "circle.png"
    cv2.imwrite(str(img_path), img)

    ordered, preview, w, h = lineart.process_image(str(img_path))

    assert len(ordered) > 0
    assert preview.shape[0] == h
    assert preview.shape[1] == w
