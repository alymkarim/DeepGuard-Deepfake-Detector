"""Vercel serverless function: score face crops with the ONNX model.

Each request carries a handful of raw RGB crops that the browser cut
out of the video. They are normalised, run through the exported
EfficientNet-B0, and returned as a fake probability per frame.

    cd web && python api/predict.py   # local dev server
"""

from __future__ import annotations

import base64
import binascii
import json
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
from flask import Flask, jsonify, request, send_from_directory

WEB_ROOT = Path(__file__).resolve().parent.parent
PUBLIC_DIR = WEB_ROOT / "public"
ASSETS_DIR = WEB_ROOT / "assets"

MODEL_PATH = Path(os.getenv("DEEPGUARD_MODEL", ASSETS_DIR / "deepguard.onnx"))
MODEL_INFO_PATH = MODEL_PATH.with_name("model_info.json")

IMAGE_SIZE = 224
MAX_FRAMES = 16
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

app = Flask(__name__)
_session: ort.InferenceSession | None = None


def get_session() -> ort.InferenceSession:
    """Load the model once per container."""

    global _session

    if _session is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Missing {MODEL_PATH.name}. Run "
                "`python scripts/export_onnx.py` at the repo root."
            )

        _session = ort.InferenceSession(
            str(MODEL_PATH),
            providers=["CPUExecutionProvider"],
        )

    return _session


def model_info() -> dict:
    if not MODEL_INFO_PATH.exists():
        return {"architecture": "efficientnet_b0", "image_size": IMAGE_SIZE}

    return json.loads(MODEL_INFO_PATH.read_text())


def decode_frames(payload: dict) -> np.ndarray:
    """Turn a list of base64 RGB crops into an NCHW float batch."""

    frames = payload.get("frames")

    if not isinstance(frames, list) or not frames:
        raise ValueError("`frames` must be a non-empty list.")

    if len(frames) > MAX_FRAMES:
        raise ValueError(f"At most {MAX_FRAMES} frames per request.")

    width = int(payload.get("width", IMAGE_SIZE))
    height = int(payload.get("height", IMAGE_SIZE))

    if width <= 0 or height <= 0 or width > IMAGE_SIZE or height > IMAGE_SIZE:
        raise ValueError(
            f"Crops must be no larger than {IMAGE_SIZE}x{IMAGE_SIZE}."
        )

    expected = width * height * 3
    batch = np.empty((len(frames), height, width, 3), dtype=np.uint8)

    for index, encoded in enumerate(frames):
        if not isinstance(encoded, str):
            raise ValueError(f"Frame {index} is not a base64 string.")

        try:
            raw = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as error:
            raise ValueError(f"Frame {index} is not valid base64.") from error

        if len(raw) != expected:
            raise ValueError(
                f"Frame {index} decoded to {len(raw)} bytes, "
                f"expected {expected}."
            )

        batch[index] = np.frombuffer(raw, dtype=np.uint8).reshape(
            height,
            width,
            3,
        )

    images = batch.astype(np.float32) / 255.0
    images = (images - MEAN) / STD

    return images.transpose(0, 3, 1, 2)


def predict(batch: np.ndarray) -> np.ndarray:
    """Return the fake probability for every image in the batch."""

    logits = get_session().run(["logits"], {"image": batch})[0]

    logits = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)

    return probabilities[:, 1]


@app.get("/api/health")
def health():
    return jsonify(
        status="ok",
        model=MODEL_PATH.name,
        exists=MODEL_PATH.exists(),
    )


@app.get("/api/model")
def model_endpoint():
    return jsonify(model_info())


@app.post("/api/predict")
def predict_endpoint():
    payload = request.get_json(silent=True)

    if payload is None:
        return jsonify(error="Expected a JSON body."), 400

    try:
        batch = decode_frames(payload)
        probabilities = predict(batch)
        faces_detected = int(payload.get("faces_detected", 0))
        sampled_frames = int(payload.get("sampled_frames", 0))
    except FileNotFoundError as error:
        return jsonify(error=str(error)), 500
    except (TypeError, ValueError) as error:
        return jsonify(error=str(error)), 400

    frame_probabilities = [round(float(value), 5) for value in probabilities]
    fake_probability = float(np.mean(probabilities))

    return jsonify(
        prediction="fake" if fake_probability >= 0.5 else "real",
        fake_probability=round(fake_probability, 5),
        real_probability=round(1.0 - fake_probability, 5),
        frame_probabilities=frame_probabilities,
        frames=len(frame_probabilities),
        faces_detected=faces_detected,
        sampled_frames=sampled_frames,
        model=model_info(),
    )


@app.get("/")
def index():
    """Served locally; Vercel answers `/` from the static layer."""

    return send_from_directory(PUBLIC_DIR, "index.html")


@app.get("/<path:filename>")
def public_file(filename: str):
    """Static assets, for local development only."""

    return send_from_directory(PUBLIC_DIR, filename)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.getenv("PORT", "5000")), debug=True)
