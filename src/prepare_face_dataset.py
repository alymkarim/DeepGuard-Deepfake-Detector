"""Extract face crops from videos into a training dataset."""

from __future__ import annotations

from pathlib import Path

import cv2

from .faces import (
    crop_face,
    load_face_detector,
    select_frame_indices,
)

IMAGE_SIZE = 224


def extract_face_crops(
    video_path: Path,
    output_directory: Path,
    frames_per_video: int = 10,
    image_size: int = IMAGE_SIZE,
    detector: cv2.CascadeClassifier | None = None,
) -> int:
    """Save evenly sampled face crops for one video.

    Returns the number of images written.
    """

    output_directory.mkdir(parents=True, exist_ok=True)

    detector = detector or load_face_detector()

    capture = cv2.VideoCapture(str(video_path))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_indices = select_frame_indices(total_frames, frames_per_video)

    if not frame_indices:
        capture.release()
        return 0

    saved = 0

    try:
        for frame_number in frame_indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

            success, frame = capture.read()

            if not success:
                continue

            crop, _ = crop_face(detector, frame, image_size)

            output_path = (
                output_directory / f"{video_path.stem}_{frame_number}.jpg"
            )

            written = cv2.imwrite(
                str(output_path),
                crop,
                [cv2.IMWRITE_JPEG_QUALITY, 95],
            )

            saved += int(written)
    finally:
        capture.release()

    return saved
