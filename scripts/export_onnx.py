"""Export the trained PyTorch checkpoint to ONNX.

The ONNX graph is what actually runs in the demos: it needs no
PyTorch, so the Vercel function and the Streamlit app can both load
the same file.

    python scripts/export_onnx.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

# Allow running this file directly as well as with `python -m`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model import create_model  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT = PROJECT_ROOT / "best_model.pth"
DEFAULT_OUTPUT = PROJECT_ROOT / "web" / "assets" / "deepguard.onnx"
INFO_OUTPUT = DEFAULT_OUTPUT.with_name("model_info.json")

CLASS_TO_IDX = {"fake": 0, "real": 1}


def load_model(checkpoint_path: Path) -> tuple[torch.nn.Module, dict]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    model = create_model(pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint


def export(model: torch.nn.Module, output_path: Path, image_size: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dummy = torch.zeros(1, 3, image_size, image_size)

    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={
            "image": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )

    onnx.checker.check_model(onnx.load(str(output_path)))


def verify(
    model: torch.nn.Module,
    onnx_path: Path,
    image_size: int,
    samples: int = 4,
    tolerance: float = 1e-3,
) -> float:
    """Compare ONNX output against PyTorch on random images."""

    torch.manual_seed(0)
    batch = torch.rand(samples, 3, image_size, image_size)

    with torch.no_grad():
        expected = torch.softmax(model(batch), dim=1).numpy()

    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )

    actual = session.run(
        ["logits"],
        {"image": batch.numpy()},
    )[0]

    actual = np.exp(actual - actual.max(axis=1, keepdims=True))
    actual = actual / actual.sum(axis=1, keepdims=True)

    error = float(np.abs(expected - actual).max())

    if error > tolerance:
        raise AssertionError(
            f"ONNX output differs from PyTorch by {error:.2e} "
            f"(tolerance {tolerance:.0e})"
        )

    return error


def write_model_info(checkpoint: dict, output_path: Path) -> None:
    info = {
        "architecture": checkpoint.get("architecture", "efficientnet_b0"),
        "image_size": checkpoint.get("image_size", 224),
        "class_to_idx": checkpoint.get("class_to_idx", CLASS_TO_IDX),
        "input_layout": "NCHW",
        "normalisation": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
    }

    metrics_path = PROJECT_ROOT / "metrics.json"

    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        info["test_metrics"] = metrics.get("test")
        info["history"] = metrics.get("history")

    output_path.write_text(json.dumps(info, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    model, checkpoint = load_model(args.checkpoint)
    image_size = checkpoint.get("image_size", 224)

    print(f"Loading {args.checkpoint}")
    print(f"Exporting to {args.output}")

    export(model, args.output, image_size)
    error = verify(model, args.output, image_size)
    print(f"Verified against PyTorch (max abs diff {error:.2e})")

    write_model_info(checkpoint, INFO_OUTPUT)
    print(f"Wrote {INFO_OUTPUT}")

    size = args.output.stat().st_size
    print(f"Model size: {size / 1_048_576:.1f} MiB")

    if shutil.which("vercel"):
        print("Run `vercel` inside web/ to deploy.")


if __name__ == "__main__":
    main()
