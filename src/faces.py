"""Face detection and cropping, shared by training and inference.

YuNet (`cv2.FaceDetectorYN`) is used instead of the classic Haar
cascades because OpenCV 5 no longer ships Haar cascades at all, and
YuNet is both faster and noticeably more accurate.

The ONNX model is vendored in ``assets/`` so detection works offline.
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

# Tuned to match the Kaggle CPU dataset builder notebook.
FACE_MARGIN_RATIO = 0.20
SCORE_THRESHOLD = 0.60

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = (
    PROJECT_ROOT / "assets" / "face_detection_yunet_2023mar.onnx"
)


def resolve_model_path() -> Path:
    """Locate the YuNet ONNX weights."""

    configured = os.getenv("FACE_DETECTOR_MODEL")

    candidates = [Path(configured)] if configured else []
    candidates.append(DEFAULT_MODEL_PATH)
    candidates.append(
        Path.cwd() / "assets" / "face_detection_yunet_2023mar.onnx"
    )

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        "The face detector model was not found. Expected it at "
        f"{DEFAULT_MODEL_PATH}. Set FACE_DETECTOR_MODEL to override."
    )


def load_face_detector(
    model_path: str | Path | None = None,
    score_threshold: float = SCORE_THRESHOLD,
) -> cv2.FaceDetectorYN:
    """Load the YuNet face detector."""

    path = Path(model_path) if model_path else resolve_model_path()

    detector = cv2.FaceDetectorYN_create(
        str(path),
        "",
        (320, 320),
        score_threshold=score_threshold,
    )

    if detector is None:
        raise RuntimeError(f"OpenCV could not load {path}")

    return detector


def select_frame_indices(
    total_frames: int,
    requested_frames: int,
) -> list[int]:
    """Evenly spaced frame positions covering the whole video."""

    if total_frames <= 0 or requested_frames <= 0:
        return []

    count = min(total_frames, requested_frames)

    return sorted(
        set(
            np.linspace(
                0,
                total_frames - 1,
                num=count,
                dtype=int,
            ).tolist()
        )
    )


def choose_largest_face(
    faces: np.ndarray | list | None,
) -> tuple[int, int, int, int] | None:
    """Return the largest rectangle from a YuNet result, or None."""

    if faces is None or len(faces) == 0:
        return None

    rectangle = max(
        faces,
        key=lambda row: float(row[2]) * float(row[3]),
    )

    return (
        int(rectangle[0]),
        int(rectangle[1]),
        int(rectangle[2]),
        int(rectangle[3]),
    )


def detect_face(
    detector: cv2.FaceDetectorYN,
    frame: np.ndarray,
) -> tuple[int, int, int, int] | None:
    """Detect the largest face in a BGR frame."""

    height, width = frame.shape[:2]
    detector.setInputSize((width, height))

    try:
        result = detector.detect(frame)
    except cv2.error:
        return None

    faces = result[1] if isinstance(result, tuple) else result

    return choose_largest_face(faces)


def expand_and_square_crop(
    frame: np.ndarray,
    face_rectangle: tuple[int, int, int, int],
    image_size: int,
    margin_ratio: float = FACE_MARGIN_RATIO,
) -> np.ndarray | None:
    """Grow a face rectangle, square it, and resize it to `image_size`."""

    x, y, width, height = face_rectangle

    frame_height, frame_width = frame.shape[:2]

    center_x = x + width // 2
    center_y = y + height // 2

    side = int(max(width, height) * (1.0 + 2.0 * margin_ratio))

    x1 = max(center_x - side // 2, 0)
    y1 = max(center_y - side // 2, 0)
    x2 = min(center_x + side // 2, frame_width)
    y2 = min(center_y + side // 2, frame_height)

    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    return cv2.resize(
        crop,
        (image_size, image_size),
        interpolation=cv2.INTER_AREA,
    )


def center_crop(
    frame: np.ndarray,
    image_size: int,
) -> np.ndarray:
    """Fallback crop when no face can be found in the frame."""

    return cv2.resize(
        frame,
        (image_size, image_size),
        interpolation=cv2.INTER_AREA,
    )


def crop_face(
    detector: cv2.FaceDetectorYN,
    frame: np.ndarray,
    image_size: int,
) -> tuple[np.ndarray, bool]:
    """Crop the largest face, falling back to the whole frame.

    Returns the crop and whether a face was actually found.
    """

    face_rectangle = detect_face(detector, frame)

    if face_rectangle is None:
        return center_crop(frame, image_size), False

    crop = expand_and_square_crop(
        frame,
        face_rectangle,
        image_size,
    )

    if crop is None:
        return center_crop(frame, image_size), False

    return crop, True
