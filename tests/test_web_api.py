"""Tests for the Vercel function in `web/api/predict.py`."""

import base64
import sys
from pathlib import Path

import numpy as np
import pytest

WEB_ROOT = Path(__file__).resolve().parent.parent / "web"
sys.path.insert(0, str(WEB_ROOT))

from api.predict import app  # noqa: E402

CROP_SIZE = 224
client = app.test_client()


def make_frame(fill: int = 0) -> str:
    frame = np.full((CROP_SIZE, CROP_SIZE, 3), fill, dtype=np.uint8)
    return base64.b64encode(frame.tobytes()).decode("ascii")


def post(payload):
    return client.post("/api/predict", json=payload)


def test_health_reports_a_usable_model():
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.get_json()["exists"] is True


def test_model_endpoint_exposes_the_reported_results():
    response = client.get("/api/model")

    assert response.status_code == 200

    body = response.get_json()

    assert body["architecture"] == "efficientnet_b0"
    assert body["image_size"] == 224
    assert body["class_to_idx"] == {"fake": 0, "real": 1}
    assert body["test_metrics"]["accuracy"] == pytest.approx(0.5133, abs=1e-4)


def test_predict_returns_a_probability_per_frame():
    response = post(
        {
            "frames": [make_frame(0), make_frame(255)],
            "width": CROP_SIZE,
            "height": CROP_SIZE,
            "faces_detected": 2,
            "sampled_frames": 4,
        }
    )

    assert response.status_code == 200

    body = response.get_json()

    assert body["prediction"] in {"fake", "real"}
    assert 0.0 <= body["fake_probability"] <= 1.0
    assert len(body["frame_probabilities"]) == 2
    assert body["frames"] == 2
    assert body["faces_detected"] == 2
    assert body["sampled_frames"] == 4
    assert body["model"]["architecture"] == "efficientnet_b0"


def test_predict_is_deterministic():
    payload = {"frames": [make_frame(128)]}

    first = post(payload).get_json()["fake_probability"]
    second = post(payload).get_json()["fake_probability"]

    assert first == second


def test_the_model_actually_reacts_to_the_input():
    black = post({"frames": [make_frame(0)]}).get_json()["fake_probability"]
    white = post({"frames": [make_frame(255)]}).get_json()["fake_probability"]

    assert black != white


def test_blanket_frame_defaults_are_accepted():
    response = post({"frames": [make_frame()]})

    assert response.status_code == 200
    assert response.get_json()["faces_detected"] == 0


def test_empty_frame_list_is_rejected():
    response = post({"frames": []})

    assert response.status_code == 400
    assert "frames" in response.get_json()["error"]


def test_missing_json_body_is_rejected():
    response = client.post("/api/predict", data="not json")

    assert response.status_code == 400


def test_wrong_frame_size_is_rejected():
    truncated = base64.b64encode(b"\x00" * 10).decode("ascii")

    response = post({"frames": [truncated]})

    assert response.status_code == 400
    assert "expected" in response.get_json()["error"]


def test_invalid_base64_is_rejected():
    response = post({"frames": ["!!! not base64 !!!"]})

    assert response.status_code == 400


def test_too_many_frames_are_rejected():
    response = post({"frames": [make_frame()] * 17})

    assert response.status_code == 400
    assert "At most" in response.get_json()["error"]


def test_oversized_crops_are_rejected():
    response = post({"frames": [make_frame()], "width": 512, "height": 512})

    assert response.status_code == 400
    assert "no larger" in response.get_json()["error"]
