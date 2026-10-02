import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import torch
from google.cloud import storage
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from .config import settings
from .dataset import make_loaders
from .evaluate import evaluate
from .gcs import upload_file, upload_text
from .model import (
    create_model,
    freeze_backbone,
    unfreeze_final_blocks,
)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
) -> dict[str, float]:
    """Train the model for one epoch."""

    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        predictions = model(images)
        loss = criterion(predictions, labels)

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size
        total_correct += (
            predictions.argmax(dim=1) == labels
        ).sum().item()
        total_samples += batch_size

    if total_samples == 0:
        raise ValueError("The training loader contains no samples.")

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
    }


def download_dataset_zip(
    bucket_name: str,
    blob_name: str,
    destination: Path,
) -> None:
    """Download the prepared dataset ZIP from Cloud Storage."""

    print(f"Downloading gs://{bucket_name}/{blob_name}")

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    if not blob.exists(client):
        raise FileNotFoundError(
            "Dataset ZIP was not found at "
            f"gs://{bucket_name}/{blob_name}"
        )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    blob.download_to_filename(str(destination))

    size_mb = destination.stat().st_size / (1024**2)

    print(f"Downloaded dataset: {size_mb:.2f} MB")


def find_dataset_root(
    extracted_directory: Path,
) -> Path:
    """
    Find the directory containing train, validation and test.

    Supported ZIP structures include:

        extracted/train/
        extracted/validation/
        extracted/test/

    or:

        extracted/deepfake-frames/train/
        extracted/deepfake-frames/validation/
        extracted/deepfake-frames/test/
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
        "Could not locate train, validation and test "
        "folders inside the extracted dataset ZIP."
    )


def count_images(directory: Path) -> int:
    """Count supported image files recursively."""

    supported_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
    }

    return sum(
        1
        for path in directory.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower() in supported_extensions
        )
    )


def validate_dataset(
    data_root: Path,
) -> dict[str, int]:
    """Validate the expected split and class folders."""

    required_directories = {
        "train_real": data_root / "train" / "real",
        "train_fake": data_root / "train" / "fake",
        "validation_real": (
            data_root / "validation" / "real"
        ),
        "validation_fake": (
            data_root / "validation" / "fake"
        ),
        "test_real": data_root / "test" / "real",
        "test_fake": data_root / "test" / "fake",
    }

    missing_directories = [
        str(directory)
        for directory in required_directories.values()
        if not directory.is_dir()
    ]

    if missing_directories:
        raise FileNotFoundError(
            "The following required dataset folders "
            "are missing:\n"
            + "\n".join(missing_directories)
        )

    counts: dict[str, int] = {}

    print("\nDataset folder counts")
    print("=" * 40)

    for name, directory in required_directories.items():
        image_count = count_images(directory)
        counts[name] = image_count

        relative_directory = directory.relative_to(
            data_root
        )

        print(
            f"{str(relative_directory):25} "
            f"{image_count:>6} images"
        )

        if image_count == 0:
            raise ValueError(
                f"No images were found in {directory}"
            )

    train_real = counts["train_real"]
    train_fake = counts["train_fake"]

    validation_real = counts["validation_real"]
    validation_fake = counts["validation_fake"]

    test_real = counts["test_real"]
    test_fake = counts["test_fake"]

    print("-" * 40)
    print(
        f"{'Train total':25} "
        f"{train_real + train_fake:>6}"
    )
    print(
        f"{'Validation total':25} "
        f"{validation_real + validation_fake:>6}"
    )
    print(
        f"{'Test total':25} "
        f"{test_real + test_fake:>6}"
    )
    print("=" * 40)

    return counts


def save_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    class_to_idx: dict[str, int],
    image_size: int,
    validation_loss: float,
    validation_roc_auc: float,
    stage_name: str,
    epoch: int,
) -> None:
    """Save the current best model checkpoint."""

    torch.save(
        {
            "architecture": "efficientnet_b0",
            "model_state_dict": model.state_dict(),
            "class_to_idx": class_to_idx,
            "image_size": image_size,
            "validation_loss": validation_loss,
            "validation_roc_auc_fake": (
                validation_roc_auc
            ),
            "training_stage": stage_name,
            "epoch": epoch,
        },
        checkpoint_path,
    )


def main() -> None:
    settings.validate()

    torch.manual_seed(42)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 60)
    print("DeepGuard training")
    print("=" * 60)
    print(f"Device: {device}")

    if device.type == "cuda":
        print(
            "GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    dataset_zip_blob = os.getenv(
        "DATASET_ZIP_BLOB",
        "datasets/deepfake-frames.zip",
    )

    with tempfile.TemporaryDirectory() as temporary_directory:
        working_directory = Path(temporary_directory)

        zip_path = (
            working_directory / "deepfake-frames.zip"
        )

        extracted_directory = (
            working_directory / "extracted"
        )

        download_dataset_zip(
            bucket_name=settings.bucket,
            blob_name=dataset_zip_blob,
            destination=zip_path,
        )

        print("\nExtracting dataset...")

        extracted_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        shutil.unpack_archive(
            filename=str(zip_path),
            extract_dir=str(extracted_directory),
        )

        data_root = find_dataset_root(
            extracted_directory
        )

        print(f"Dataset root: {data_root}")

        folder_counts = validate_dataset(data_root)

        (
            train_loader,
            validation_loader,
            test_loader,
            class_to_idx,
        ) = make_loaders(
            data_root,
            settings.image_size,
            settings.batch_size,
            settings.num_workers,
        )

        print("\nDataLoader dataset sizes")
        print("=" * 40)
        print(
            f"Train images: "
            f"{len(train_loader.dataset)}"
        )
        print(
            f"Validation images: "
            f"{len(validation_loader.dataset)}"
        )
        print(
            f"Test images: "
            f"{len(test_loader.dataset)}"
        )
        print(f"Class mapping: {class_to_idx}")
        print("=" * 40)

        if "fake" not in class_to_idx:
            raise ValueError(
                "The dataset must contain a class "
                "folder named 'fake'."
            )

        if "real" not in class_to_idx:
            raise ValueError(
                "The dataset must contain a class "
                "folder named 'real'."
            )

        fake_class_index = class_to_idx["fake"]

        model = create_model(
            pretrained=True
        ).to(device)

        freeze_backbone(model)

        criterion = nn.CrossEntropyLoss()

        checkpoint_path = (
            working_directory / "best_model.pth"
        )

        history: list[dict[str, Any]] = []

        best_overall_validation_loss = float("inf")
        best_overall_roc_auc = 0.0

        def run_training_stage(
            number_of_epochs: int,
            learning_rate: float,
            stage_name: str,
            early_stopping_patience: int = 3,
        ) -> None:
            nonlocal best_overall_validation_loss
            nonlocal best_overall_roc_auc

            trainable_parameters = [
                parameter
                for parameter in model.parameters()
                if parameter.requires_grad
            ]

            if not trainable_parameters:
                raise ValueError(
                    f"No trainable parameters were found "
                    f"for stage '{stage_name}'."
                )

            optimizer = torch.optim.AdamW(
                trainable_parameters,
                lr=learning_rate,
                weight_decay=1e-4,
            )

            scheduler = (
                torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode="min",
                    factor=0.5,
                    patience=1,
                    min_lr=1e-7,
                )
            )

            stage_best_validation_loss = float("inf")
            epochs_without_improvement = 0

            print("\n" + "=" * 60)
            print(
                f"Starting stage: {stage_name}"
            )
            print(
                f"Maximum epochs: {number_of_epochs}"
            )
            print(
                f"Initial learning rate: "
                f"{learning_rate}"
            )
            print(
                f"Early-stopping patience: "
                f"{early_stopping_patience}"
            )
            print("=" * 60)

            for epoch_index in range(number_of_epochs):
                epoch_number = epoch_index + 1

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

                validation_loss = float(
                    validation_metrics["loss"]
                )

                validation_roc_auc = float(
                    validation_metrics.get(
                        "roc_auc_fake",
                        0.0,
                    )
                )

                scheduler.step(validation_loss)

                current_learning_rate = (
                    optimizer.param_groups[0]["lr"]
                )

                epoch_result = {
                    "stage": stage_name,
                    "epoch": epoch_number,
                    "learning_rate": (
                        current_learning_rate
                    ),
                    "train": training_metrics,
                    "validation": validation_metrics,
                }

                history.append(epoch_result)

                print("\n" + "-" * 60)
                print(
                    f"{stage_name} | "
                    f"Epoch {epoch_number}/"
                    f"{number_of_epochs}"
                )
                print("-" * 60)
                print(
                    json.dumps(
                        epoch_result,
                        indent=2,
                    )
                )

                stage_improved = (
                    validation_loss
                    < stage_best_validation_loss
                )

                if stage_improved:
                    stage_best_validation_loss = (
                        validation_loss
                    )

                    epochs_without_improvement = 0

                    print(
                        "Stage validation loss improved "
                        f"to {validation_loss:.4f}"
                    )
                else:
                    epochs_without_improvement += 1

                    print(
                        "Stage validation loss did not "
                        "improve."
                    )
                    print(
                        "Early-stopping counter: "
                        f"{epochs_without_improvement}/"
                        f"{early_stopping_patience}"
                    )

                overall_improved = (
                    validation_loss
                    < best_overall_validation_loss
                )

                if overall_improved:
                    best_overall_validation_loss = (
                        validation_loss
                    )

                    best_overall_roc_auc = (
                        validation_roc_auc
                    )

                    save_checkpoint(
                        model=model,
                        checkpoint_path=checkpoint_path,
                        class_to_idx=class_to_idx,
                        image_size=settings.image_size,
                        validation_loss=validation_loss,
                        validation_roc_auc=(
                            validation_roc_auc
                        ),
                        stage_name=stage_name,
                        epoch=epoch_number,
                    )

                    print(
                        "Saved new overall best model."
                    )
                    print(
                        "Best validation loss: "
                        f"{best_overall_validation_loss:.4f}"
                    )
                    print(
                        "Validation ROC AUC: "
                        f"{best_overall_roc_auc:.4f}"
                    )

                if (
                    epochs_without_improvement
                    >= early_stopping_patience
                ):
                    print(
                        f"Early stopping triggered during "
                        f"'{stage_name}'."
                    )
                    break

        run_training_stage(
            number_of_epochs=settings.epochs,
            learning_rate=settings.learning_rate,
            stage_name="classifier",
            early_stopping_patience=3,
        )

        print("\nUnfreezing final model blocks...")

        unfreeze_final_blocks(model)

        run_training_stage(
            number_of_epochs=(
                settings.fine_tune_epochs
            ),
            learning_rate=(
                settings.fine_tune_learning_rate
            ),
            stage_name="fine_tune",
            early_stopping_patience=3,
        )

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                "Training completed without producing "
                "a model checkpoint."
            )

        print("\nLoading best checkpoint...")

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        print(
            "Best checkpoint came from "
            f"{checkpoint.get('training_stage')} "
            f"epoch {checkpoint.get('epoch')}."
        )

        print("\nEvaluating best model on test set...")

        test_metrics = evaluate(
            model,
            test_loader,
            criterion,
            device,
            fake_class_index,
        )

        results = {
            "dataset": {
                "folder_counts": folder_counts,
                "train_size": len(
                    train_loader.dataset
                ),
                "validation_size": len(
                    validation_loader.dataset
                ),
                "test_size": len(
                    test_loader.dataset
                ),
            },
            "training_configuration": {
                "architecture": "efficientnet_b0",
                "device": str(device),
                "image_size": settings.image_size,
                "batch_size": settings.batch_size,
                "classifier_epochs": settings.epochs,
                "fine_tune_epochs": (
                    settings.fine_tune_epochs
                ),
                "classifier_learning_rate": (
                    settings.learning_rate
                ),
                "fine_tune_learning_rate": (
                    settings.fine_tune_learning_rate
                ),
            },
            "best_checkpoint": {
                "stage": checkpoint.get(
                    "training_stage"
                ),
                "epoch": checkpoint.get("epoch"),
                "validation_loss": checkpoint.get(
                    "validation_loss"
                ),
                "validation_roc_auc_fake": (
                    checkpoint.get(
                        "validation_roc_auc_fake"
                    )
                ),
            },
            "history": history,
            "test": test_metrics,
            "class_to_idx": class_to_idx,
        }

        metrics_path = (
            working_directory / "metrics.json"
        )

        metrics_path.write_text(
            json.dumps(results, indent=2),
            encoding="utf-8",
        )

        print("\nUploading training artifacts...")

        upload_file(
            settings.bucket,
            checkpoint_path,
            (
                f"{settings.output_prefix}/"
                "best_model.pth"
            ),
        )

        upload_file(
            settings.bucket,
            metrics_path,
            (
                f"{settings.output_prefix}/"
                "metrics.json"
            ),
        )

        upload_text(
            settings.bucket,
            json.dumps(
                class_to_idx,
                indent=2,
            ),
            (
                f"{settings.output_prefix}/"
                "class_to_idx.json"
            ),
        )

        print("\n" + "=" * 60)
        print("Training completed successfully")
        print("=" * 60)
        print(
            json.dumps(
                test_metrics,
                indent=2,
            )
        )
        print(
            "\nModel uploaded to:\n"
            f"gs://{settings.bucket}/"
            f"{settings.output_prefix}/"
            "best_model.pth"
        )
        print(
            "\nMetrics uploaded to:\n"
            f"gs://{settings.bucket}/"
            f"{settings.output_prefix}/"
            "metrics.json"
        )


if __name__ == "__main__":
    main()