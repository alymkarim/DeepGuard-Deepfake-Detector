import json
import os
import shutil
import tempfile
from pathlib import Path

import torch
from google.cloud import storage
from torch import nn

from .config import settings
from .dataset import make_loaders
from .evaluate import evaluate
from .gcs import upload_file, upload_text
from .model import create_model, freeze_backbone, unfreeze_final_blocks


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        predictions = model(images)
        loss = criterion(predictions, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_correct += (
            predictions.argmax(dim=1) == labels
        ).sum().item()
        total_samples += labels.size(0)

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
    }


def download_dataset_zip(
    bucket_name: str,
    blob_name: str,
    destination: Path,
) -> None:
    """Download the frame dataset ZIP from Cloud Storage."""

    print(f"Downloading gs://{bucket_name}/{blob_name}")

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    if not blob.exists(client):
        raise FileNotFoundError(
            f"Dataset was not found at "
            f"gs://{bucket_name}/{blob_name}"
        )

    blob.download_to_filename(str(destination))

    size_mb = destination.stat().st_size / (1024**2)
    print(f"Downloaded dataset: {size_mb:.2f} MB")


def find_dataset_root(extracted_directory: Path) -> Path:
    """
    Find the folder containing train, validation and test.

    Supports ZIP structures such as:
        extracted/train/
        extracted/deepfake-frames/train/
    """

    candidates = [extracted_directory]

    candidates.extend(
        path
        for path in extracted_directory.rglob("*")
        if path.is_dir()
    )

    for candidate in candidates:
        train_directory = candidate / "train"
        validation_directory = candidate / "validation"
        test_directory = candidate / "test"

        if (
            train_directory.is_dir()
            and validation_directory.is_dir()
            and test_directory.is_dir()
        ):
            return candidate

    raise FileNotFoundError(
        "Could not locate train, validation and test folders "
        "inside the extracted ZIP."
    )


def validate_dataset(data_root: Path) -> None:
    """Check that every required split and class exists."""

    required_directories = [
        data_root / "train" / "real",
        data_root / "train" / "fake",
        data_root / "validation" / "real",
        data_root / "validation" / "fake",
        data_root / "test" / "real",
        data_root / "test" / "fake",
    ]

    missing = [
        str(path)
        for path in required_directories
        if not path.is_dir()
    ]

    if missing:
        raise FileNotFoundError(
            "The following required dataset folders are missing:\n"
            + "\n".join(missing)
        )

    for directory in required_directories:
        image_count = len(list(directory.rglob("*.jpg")))
        image_count += len(list(directory.rglob("*.jpeg")))
        image_count += len(list(directory.rglob("*.png")))

        print(
            f"{directory.relative_to(data_root)}: "
            f"{image_count} images"
        )

        if image_count == 0:
            raise ValueError(
                f"No images were found in {directory}"
            )


def main():
    settings.validate()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    # Can be overridden when submitting the Vertex AI job.
    dataset_zip_blob = os.getenv(
        "DATASET_ZIP_BLOB",
        "datasets/deepfake-frames.zip",
    )

    with tempfile.TemporaryDirectory() as temporary_directory:
        working_directory = Path(temporary_directory)

        zip_path = working_directory / "deepfake-frames.zip"
        extracted_directory = working_directory / "extracted"

        download_dataset_zip(
            bucket_name=settings.bucket,
            blob_name=dataset_zip_blob,
            destination=zip_path,
        )

        print("Extracting dataset...")
        extracted_directory.mkdir(parents=True, exist_ok=True)

        shutil.unpack_archive(
            filename=str(zip_path),
            extract_dir=str(extracted_directory),
        )

        data_root = find_dataset_root(extracted_directory)

        print(f"Dataset root: {data_root}")

        validate_dataset(data_root)

        train_loader, validation_loader, test_loader, class_to_idx = (
            make_loaders(
                data_root,
                settings.image_size,
                settings.batch_size,
                settings.num_workers,
            )
        )

        print(f"Class mapping: {class_to_idx}")

        if "fake" not in class_to_idx:
            raise ValueError(
                "The dataset must contain a class folder named 'fake'."
            )

        fake_class_index = class_to_idx["fake"]

        model = create_model(pretrained=True).to(device)
        freeze_backbone(model)

        criterion = nn.CrossEntropyLoss()

        best_validation_loss = float("inf")
        checkpoint_path = working_directory / "best_model.pth"
        history = []

        def run_training_stage(
            number_of_epochs: int,
            learning_rate: float,
            stage_name: str,
        ):
            nonlocal best_validation_loss

            optimizer = torch.optim.AdamW(
                filter(
                    lambda parameter: parameter.requires_grad,
                    model.parameters(),
                ),
                lr=learning_rate,
                weight_decay=1e-4,
            )

            for epoch_index in range(number_of_epochs):
                training_metrics = train_epoch(
                    model=model,
                    loader=train_loader,
                    criterion=criterion,
                    optimizer=optimizer,
                    device=device,
                )

                validation_metrics = evaluate(
                    model,
                    validation_loader,
                    criterion,
                    device,
                    fake_class_index,
                )

                epoch_result = {
                    "stage": stage_name,
                    "epoch": epoch_index + 1,
                    "train": training_metrics,
                    "validation": validation_metrics,
                }

                history.append(epoch_result)
                print(json.dumps(epoch_result, indent=2))

                if (
                    validation_metrics["loss"]
                    < best_validation_loss
                ):
                    best_validation_loss = validation_metrics["loss"]

                    torch.save(
                        {
                            "architecture": "efficientnet_b0",
                            "model_state_dict": model.state_dict(),
                            "class_to_idx": class_to_idx,
                            "image_size": settings.image_size,
                        },
                        checkpoint_path,
                    )

                    print(
                        "Saved new best checkpoint with "
                        f"validation loss "
                        f"{best_validation_loss:.4f}"
                    )

        run_training_stage(
            number_of_epochs=settings.epochs,
            learning_rate=settings.learning_rate,
            stage_name="classifier",
        )

        unfreeze_final_blocks(model)

        run_training_stage(
            number_of_epochs=settings.fine_tune_epochs,
            learning_rate=settings.fine_tune_learning_rate,
            stage_name="fine_tune",
        )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        test_metrics = evaluate(
            model,
            test_loader,
            criterion,
            device,
            fake_class_index,
        )

        results = {
            "history": history,
            "test": test_metrics,
            "class_to_idx": class_to_idx,
        }

        metrics_path = working_directory / "metrics.json"
        metrics_path.write_text(
            json.dumps(results, indent=2),
            encoding="utf-8",
        )

        upload_file(
            settings.bucket,
            checkpoint_path,
            f"{settings.output_prefix}/best_model.pth",
        )

        upload_file(
            settings.bucket,
            metrics_path,
            f"{settings.output_prefix}/metrics.json",
        )

        upload_text(
            settings.bucket,
            json.dumps(class_to_idx, indent=2),
            f"{settings.output_prefix}/class_to_idx.json",
        )

        print("Training completed.")
        print(json.dumps(test_metrics, indent=2))
        print(
            f"Model uploaded to "
            f"gs://{settings.bucket}/"
            f"{settings.output_prefix}/best_model.pth"
        )


if __name__ == "__main__":
    main()