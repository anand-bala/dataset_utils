#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import NamedTuple
import subprocess as sp

from dataset_utils import rm_imgs_without_labels

LABEL_MAP = {
    "car": 0,
    "bus": 1,
    "person": 2,
    "bike": 3,
    "truck": 4,
    "motor": 5,
    "train": 6,
    "rider": 7,
    "traffic sign": 8,
    "traffic light": 9,
}

IMG_WIDTH = 1280
IMG_HEIGHT = 720


class Arguments(NamedTuple):
    data_path: Path
    train_labels: Path
    val_labels: Path
    output_dir: Path


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="""
            Create a dataset of images and labels, along with a corresponding
            bdd100k.data file, a train.txt, and a validation.txt that can be inputed
            into darknet to train a YOLO model on the BDD100k dataset.

            WARNING: This will copy the images in the dataset to a different directory.
            I am OK with this as storage is cheap on my PC, but modify this if you don't
            like it.
        """
    )

    ap.add_argument(
        "--data-path",
        help="Path to BDD dataset root (e.g. bdd100k/images/100k). Should contain the directories `train`, `test`, and `val` with .jpg images",
    )

    ap.add_argument(
        "--train-labels",
        help="Path to BDD100k training labels JSON file (e.g. bdd100k_labels_images_train.json)",
    )

    ap.add_argument(
        "--val-labels",
        help="Path to BDD100k validation labels JSON file (e.g. bdd100k_labels_images_val.json)",
    )

    ap.add_argument(
        "--output-dir",
        help="Path to output the YOLO compatible dataset and other darknet helper files",
    )

    return ap.parse_args()


def validate_args(args: argparse.Namespace) -> Arguments:
    data_path = Path(args.data_path).absolute().resolve()
    assert data_path.is_dir(), "Given data path is not a directory"
    assert (
        data_path / "train"
    ).is_dir(), "Given data path doesn't contain a subdirectory `train`"
    assert (
        data_path / "val"
    ).is_dir(), "Given data path doesn't contain a subdirectory `val`"
    assert (
        data_path / "test"
    ).is_dir(), "Given data path doesn't contain a subdirectory `test`"

    train_labels = Path(args.train_labels).absolute().resolve()
    assert (
        train_labels.is_file()
    ), "Given training labels path is either not a file or doesn't exist"
    val_labels = Path(args.val_labels).absolute().resolve()
    assert (
        val_labels.is_file()
    ), "Given validation labels path is either not a file or doesn't exist"

    output_dir = Path(args.output_dir).absolute().resolve()
    if output_dir.is_dir():
        import sys

        print(
            "[WARNING] Output directory already exists, contents may be overwritten",
            file=sys.stderr,
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    return Arguments(
        data_path=data_path,
        train_labels=train_labels,
        val_labels=val_labels,
        output_dir=output_dir,
    )


def box2d_to_yolo(box2d):
    x1 = box2d["x1"] / IMG_WIDTH
    x2 = box2d["x2"] / IMG_WIDTH
    y1 = box2d["y1"] / IMG_HEIGHT
    y2 = box2d["y2"] / IMG_HEIGHT

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    width = abs(x2 - x1)
    height = abs(y2 - y1)

    return cx, cy, width, height


def label2txt(labels_json: Path, output_dir: Path):
    """
    This function converts the labels into a .txt file with the same name as the image.
    It extracts the bounding box, class info from the .json file and converts it into
    the darknet format.

    The darknet format is
        <object id> <x> <y> <width> <height>
    """
    assert labels_json.is_file(), "Labels JSON file doesn't exist"
    assert output_dir.is_dir(), "Output directory doesn't exist"
    frames = json.load(open(labels_json, "r"))

    for frame in frames:
        img_name = Path(frame["name"])
        assert img_name.suffix == ".jpg"
        frame_file = output_dir / (img_name.with_suffix(".txt"))

        # Creates, opens, and adds to a txt file with the name of each image.jpg
        with open(frame_file, "w+") as f:
            # For each sub label of each image, get the box2d variable
            # Get the relative center point compared to the image size 1280/720
            for label in frame["labels"]:
                if "box2d" not in label:
                    continue
                box2d = label["box2d"]
                if box2d["x1"] >= box2d["x2"] or box2d["y1"] >= box2d["y2"]:
                    continue
                cx, cy, width, height = box2d_to_yolo(box2d)
                lbl = LABEL_MAP[label["category"]]

                f.write("{} {} {} {} {}\n".format(lbl, cx, cy, width, height))


if __name__ == "__main__":
    args = validate_args(parse_args())

    # First, copy each data directory over to the output directory.
    for dir in ["train", "val", "test"]:
        src = args.data_path / dir
        dst = args.output_dir / dir
        dst.mkdir(parents=True, exist_ok=True)
        cp_cmd = [
            "rsync",
            "-a",
            str(src) + "/", # Trailing slash needed for rsync
            str(dst),
        ]
        print("-- Copying the data over to {}".format(dst))
        print("> {}".format(" ".join(cp_cmd)))
        proc = sp.Popen(cp_cmd, stdout=sp.DEVNULL)
        if dir == "train" or dir == "val":
            print("-- Generating labels at that dir in parallel")
        if dir == "train":
            label2txt(args.train_labels, dst)
        if dir == "val":
            label2txt(args.val_labels, dst)

        proc.wait()
        print("-- Done copying")

        if dir == "train" or dir == "val":
            print("-- Removing images without corresponding labels")
            rm_imgs_without_labels(dst)

    # Create names file
    names = [''] * len(LABEL_MAP)
    for label, num in LABEL_MAP.items():
        names[num] = label
    names_file = args.output_dir / "bdd100k.names"
    with open(names_file, "w+") as f:
        f.write("\n".join(names))

