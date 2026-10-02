from __future__ import annotations

from pathlib import Path

import cv2
from PIL import Image
from torchvision import transforms

from .faces import load_face_detector, crop_face, select_frame_indices
from .model import create_model

import torch


class VideoDeepfakePredictor:
    """Score a video by classifying evenly sampled face crops."""

    def __init__(self, checkpoint_path: str | Path):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        checkpoint = torch.load(
            str(checkpoint_path),
            map_location=self.device,
            weights_only=False,
        )

        self.class_to_idx = checkpoint["class_to_idx"]
        self.fake_index = self.class_to_idx["fake"]
        self.image_size = checkpoint.get("image_size", 224)
        self.checkpoint_metadata = {
            key: checkpoint.get(key)
            for key in (
                "architecture",
                "training_stage",
                "epoch",
                "validation_loss",
                "validation_roc_auc_fake",
            )
        }

        self.model = create_model(pretrained=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self.face_detector = load_face_detector()

        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def predict(
        self,
        video_path: str | Path,
        frames_to_sample: int = 12,
    ) -> dict:
        capture = cv2.VideoCapture(str(video_path))

        if not capture.isOpened():
            raise ValueError(
                f"Could not open video: {video_path}"
            )

        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_indices = select_frame_indices(
            total_frames,
            frames_to_sample,
        )

        probabilities: list[float] = []
        faces_detected = 0

        try:
            for frame_number in frame_indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

                success, frame = capture.read()

                if not success:
                    continue

                crop, found_face = crop_face(
                    self.face_detector,
                    frame,
                    self.image_size,
                )

                faces_detected += int(found_face)

                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                tensor = (
                    self.transform(Image.fromarray(rgb))
                    .unsqueeze(0)
                    .to(self.device)
                )

                with torch.no_grad():
                    logits = self.model(tensor)
                    probability = torch.softmax(logits, dim=1)[
                        0,
                        self.fake_index,
                    ]

                probabilities.append(float(probability.item()))
        finally:
            capture.release()

        if not probabilities:
            raise ValueError(
                "No usable frames were found in the video."
            )

        fake_probability = sum(probabilities) / len(probabilities)

        return {
            "prediction": "fake" if fake_probability >= 0.5 else "real",
            "fake_probability": fake_probability,
            "real_probability": 1 - fake_probability,
            "frames_used": len(probabilities),
            "faces_detected": faces_detected,
            "sampled_frames": len(frame_indices),
        }
