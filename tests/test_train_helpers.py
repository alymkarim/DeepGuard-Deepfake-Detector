import json

import pytest
import torch
from torch import nn

from src.model import create_model, freeze_backbone, unfreeze_final_blocks
from src.train import (
    count_images,
    find_dataset_root,
    save_checkpoint,
    validate_dataset,
)


def make_split_tree(root):
    for split in ("train", "validation", "test"):
        for label in ("real", "fake"):
            directory = root / split / label
            directory.mkdir(parents=True, exist_ok=True)
            for index in range(2):
                (directory / f"frame_{index}.jpg").write_bytes(b"fake")
    return root


def test_find_dataset_root_handles_a_nested_archive(tmp_path):
    nested = tmp_path / "extracted" / "deepfake-frames"
    make_split_tree(nested)

    assert find_dataset_root(tmp_path / "extracted") == nested


def test_find_dataset_root_raises_when_splits_are_missing(tmp_path):
    (tmp_path / "train").mkdir()

    with pytest.raises(FileNotFoundError):
        find_dataset_root(tmp_path)


def test_count_images_ignores_unsupported_files(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"a")
    (tmp_path / "b.png").write_bytes(b"b")
    (tmp_path / "notes.txt").write_text("no")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "c.webp").write_bytes(b"c")

    assert count_images(tmp_path) == 3


def test_validate_dataset_reports_every_split(tmp_path):
    root = make_split_tree(tmp_path)

    counts = validate_dataset(root)

    assert counts == {
        "train_real": 2,
        "train_fake": 2,
        "validation_real": 2,
        "validation_fake": 2,
        "test_real": 2,
        "test_fake": 2,
    }


def test_validate_dataset_rejects_an_empty_class_folder(tmp_path):
    root = make_split_tree(tmp_path)

    for path in (root / "train" / "fake").iterdir():
        path.unlink()

    with pytest.raises(ValueError):
        validate_dataset(root)


def test_validate_dataset_reports_missing_folders(tmp_path):
    with pytest.raises(FileNotFoundError):
        validate_dataset(tmp_path)


def test_save_checkpoint_writes_the_inference_contract(tmp_path):
    model = nn.Linear(4, 2)
    checkpoint_path = tmp_path / "best_model.pth"

    save_checkpoint(
        model=model,
        checkpoint_path=checkpoint_path,
        class_to_idx={"fake": 0, "real": 1},
        image_size=224,
        validation_loss=0.42,
        validation_roc_auc=0.91,
        stage_name="fine_tune",
        epoch=3,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    assert checkpoint["class_to_idx"] == {"fake": 0, "real": 1}
    assert checkpoint["image_size"] == 224
    assert checkpoint["architecture"] == "efficientnet_b0"
    assert checkpoint["training_stage"] == "fine_tune"
    assert checkpoint["epoch"] == 3
    assert checkpoint["validation_loss"] == 0.42
    assert checkpoint["validation_roc_auc_fake"] == 0.91
    assert "model_state_dict" in checkpoint


def test_checkpoint_round_trips_through_the_demo_model(tmp_path):
    """The weights saved by training must load in the demo model."""

    trained = create_model(pretrained=False)
    checkpoint_path = tmp_path / "best_model.pth"

    save_checkpoint(
        model=trained,
        checkpoint_path=checkpoint_path,
        class_to_idx={"fake": 0, "real": 1},
        image_size=224,
        validation_loss=0.5,
        validation_roc_auc=0.5,
        stage_name="classifier",
        epoch=1,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    inference_model = create_model(pretrained=False)
    inference_model.load_state_dict(checkpoint["model_state_dict"])

    assert json.loads(json.dumps(checkpoint["class_to_idx"])) == {
        "fake": 0,
        "real": 1,
    }


def test_freeze_helpers_control_the_backbone():
    model = create_model(pretrained=False)

    freeze_backbone(model)

    assert not any(
        parameter.requires_grad
        for parameter in model.features.parameters()
    )
    assert model.classifier[1].weight.requires_grad

    unfreeze_final_blocks(model, n=2)

    assert any(
        parameter.requires_grad
        for parameter in model.features[-2:].parameters()
    )
