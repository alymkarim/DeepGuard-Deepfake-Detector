import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

use_pin_memory = torch.cuda.is_available()
def make_loaders(
    data_root,
    image_size,
    batch_size,
    num_workers,
):
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.10,
        ),
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=3)],
            p=0.15,
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    evaluation_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    train_dataset = datasets.ImageFolder(
        data_root / "train",
        transform=train_transform,
    )

    validation_dataset = datasets.ImageFolder(
        data_root / "validation",
        transform=evaluation_transform,
    )

    test_dataset = datasets.ImageFolder(
        data_root / "test",
        transform=evaluation_transform,
    )

    print(f"Train images: {len(train_dataset)}")
    print(f"Validation images: {len(validation_dataset)}")
    print(f"Test images: {len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )

    return (
        train_loader,
        validation_loader,
        test_loader,
        train_dataset.class_to_idx,
    )