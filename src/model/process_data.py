#!/usr/bin/env python
import argparse
import pathlib
import random
import sys

import cv2
import numpy as np
import torch
from torchvision import transforms
from torchvision import datasets
import matplotlib.pyplot as plt


SCRIPT_NAME = pathlib.Path(__name__).name

LABEL_MAP = {0: "Glaucoma not Present", 1: "Glaucoma Present"}
# ImageNet means and std.
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def parse_args() -> argparse.Namespace:
    """Returns a namespace of parsed command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data_directory", type=str, help="directory with train, test and val datasets"
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        help="directory to which processed datasets should be written.",
    )
    parser.add_argument(
        "--size",
        action="store_true",
        help="calculate the size of the dataset and exit.",
    )
    parser.add_argument(
        "--save-visualizations",
        action="store_true",
        help="store visualization under the default filename in `output-dir` or in current directory",
    )
    return parser.parse_args()


def load_image(path: pathlib.Path) -> torch.Tensor:
    """Returns a PyTorch tensor representing the image from `path`."""
    if not path.exists():
        raise FileNotFoundError("No such file or directory")
    if path.is_dir():
        raise ValueError(f"{path} is a  directory")
    return torch.Tensor(cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB))


def data_size(path: pathlib.Path, verbose: bool = False) -> int:
    """Returns size of dataset in bytes. Slow, naïve calculation of dataset size."""
    total = 0
    for img_path in path.glob("./**/*.png"):
        img = np.array(load_image(img_path))
        total += img.nbytes
    if verbose:
        print(
            "Total size of the dataset:"
            f"{total / 2 ** 10} kiB, {total / 2 ** 20} MiB, {total / 2 ** 30} GiB"
        )
    return total


def mean_and_std(path: pathlib.Path, sample_size: int = 1000):
    """Returns the mean and std of images in dataset at `path`.

    The values are calculated based on `sample_size` sample images from dataset.
    To use all images from the dataset to calculate those values pass -1, as
    `sample_size`."""
    if sample_size < 0:
        sample_size = len(list(path.glob("./data/**/*.png")))
    sample = random.sample(list(path.glob("./**/*.png")), k=sample_size)
    images = torch.stack([load_image(file) for file in sample])
    return images.mean(dim=(0, 1, 2)), images.std(dim=(0, 1, 2))


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    return np.transpose(
        (tensor - tensor.min()) * 1 / (tensor.max() - tensor.min()), (1, 2, 0)
    )


def setup_data_loaders(data: pathlib.Path) -> dict[str, datasets.ImageFolder]:
    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.RandomAutocontrast(1),
            transforms.Normalize(mean=MEAN, std=STD),
        ]
    )
    train_loader = datasets.ImageFolder(str(data / "train"), transform=transform)
    test_loader = datasets.ImageFolder(str(data / "test"), transform=transform)
    val_loader = datasets.ImageFolder(str(data / "val"), transform=transform)

    return {"train": train_loader, "val": val_loader, "test": test_loader}


def main() -> int:
    args = parse_args()
    data = pathlib.Path(args.data_directory)

    # Exit if passed data directory doesn't exist
    if not data.exists():
        print(f"{SCRIPT_NAME}: error: No such file or directory")
        return 1

    # if `--size` option was passed, print size of the dataset and exit.
    if args.size:
        data_size(data, verbose=True)
        return 0

    loaders = setup_data_loaders(data)

    for name, loader in loaders.items():
        sample_idx = torch.randint(len(loader), size=(1,))
        rows, cols = 2, 6
        fig = plt.figure(figsize=(18, 6))
        for i in range(1, rows * cols + 1):
            sample_idx = random.randint(0, len(loader))
            img, label = loader[sample_idx]
            img = denormalize(img)
            fig.add_subplot(rows, cols, i)
            fig.suptitle(f"{name.capitalize()} dataset")
            plt.title(LABEL_MAP[label])
            plt.axis("off")
            plt.imshow(img)

    if args.output_dir:
        output = pathlib.Path(args.output_dir)
        if not output.exists():
            print(f"{SCRIPT_NAME}: error: No such file or directory")
            return 1
        if not output.is_dir():
            print(f"{SCRIPT_NAME}: warning: {output} is not a directory.", end=" ")
            return 1

        for name, loader in loaders.items():
            save_path = output / f"{name}.pt"
            if save_path.exists():
                print(f"{SCRIPT_NAME}: warning: File {save_path} exists.", end=" ")
                choice = input("Do you wish to overwrite it? [Y/n]: ")
                if choice.lower() in ["y", "yes"]:
                    torch.save(loader, output)
                else:
                    return 1
            else:
                torch.save(loader, save_path)

        if args.save_visualizations:
            for i in plt.get_fignums():
                plt.figure(i)
                plt.savefig(output / f"Figure{i}.png")

    if args.save_visualizations and not args.output_dir:
        for i in plt.get_fignums():
            plt.figure(i)
            plt.savefig(f"./Figure{i}.png")

    plt.tight_layout()
    plt.show()

    return 0


if __name__ == "__main__":
    sys.exit(main())
