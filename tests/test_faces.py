import cv2
import numpy as np
import pytest

from src.faces import (
    center_crop,
    choose_largest_face,
    crop_face,
    detect_face,
    expand_and_square_crop,
    load_face_detector,
    resolve_model_path,
    select_frame_indices,
)


def test_select_frame_indices_covers_the_whole_video():
    indices = select_frame_indices(total_frames=100, requested_frames=5)

    assert indices == [0, 24, 49, 74, 99]
    assert len(indices) == 5


def test_select_frame_indices_is_capped_by_available_frames():
    indices = select_frame_indices(total_frames=3, requested_frames=12)

    assert indices == [0, 1, 2]


def test_select_frame_indices_handles_missing_metadata():
    assert select_frame_indices(0, 12) == []
    assert select_frame_indices(-5, 12) == []
    assert select_frame_indices(100, 0) == []


def test_choose_largest_face_picks_the_biggest_rectangle():
    faces = [(10, 10, 20, 20), (0, 0, 50, 50), (5, 5, 30, 10)]

    assert choose_largest_face(faces) == (0, 0, 50, 50)


def test_choose_largest_face_handles_no_detections():
    assert choose_largest_face([]) is None
    assert choose_largest_face(None) is None


def test_expand_and_square_crop_keeps_the_face_in_frame():
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    crop = expand_and_square_crop(frame, (90, 40, 30, 30), image_size=64)

    assert crop.shape == (64, 64, 3)


def test_expand_and_square_crop_never_reads_outside_the_frame():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    crop = expand_and_square_crop(frame, (0, 0, 10, 10), image_size=32)

    assert crop.shape == (32, 32, 3)


def test_center_crop_resizes_to_the_model_input_size():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    assert center_crop(frame, 224).shape == (224, 224, 3)


def test_load_face_detector_returns_usable_yunet():
    detector = load_face_detector()

    assert isinstance(detector, cv2.FaceDetectorYN)


def test_resolve_model_path_finds_the_vendored_weights():
    assert resolve_model_path().name.endswith(".onnx")
    assert resolve_model_path().is_file()


def test_detect_face_returns_none_on_an_empty_frame():
    detector = load_face_detector()
    frame = np.zeros((300, 300, 3), dtype=np.uint8)

    assert detect_face(detector, frame) is None


def test_crop_face_falls_back_to_the_whole_frame():
    detector = load_face_detector()
    frame = np.zeros((300, 300, 3), dtype=np.uint8)

    crop, found_face = crop_face(detector, frame, image_size=64)

    assert crop.shape == (64, 64, 3)
    assert found_face is False


@pytest.mark.parametrize("image_size", [64, 224])
def test_crop_face_returns_the_model_input_shape(image_size):
    detector = load_face_detector()

    # A gradient gives the detector something to work with while
    # still being unlikely to look like a face.
    frame = np.tile(
        np.linspace(0, 255, 400, dtype=np.uint8).reshape(1, -1),
        (400, 1),
    )
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    crop, _ = crop_face(detector, frame, image_size=image_size)

    assert crop.shape == (image_size, image_size, 3)
