#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
from typing import NamedTuple, List
import subprocess as sp

from dataset_utils import rm_imgs_without_labels

LABEL_MAP = {
    "Car": 0,
    "Van": 1,
    "Truck": 2,
    "Pedestrian": 3,
    "Person_sitting": 4,
    "Cyclist": 5,
    "Tram": 6,
    "Misc": 7,
}

IMG_WIDTH = 1224
IMG_HEIGHT = 370


class Arguments(NamedTuple):
    data_path: Path
    output_dir: Path


class BoundingBox(NamedTuple):
    label: str
    left: float
    right: float
    top: float
    bottom: float


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="""
            Create a dataset of images and labels, along with a corresponding
            kitti.data file, a train.txt, and a validation.txt that can be inputed
            into darknet to train a YOLO model on the KITTI dataset.

            WARNING: This will copy the images in the dataset to a different directory.
            I am OK with this as storage is cheap on my PC, but modify this if you don't
            like it.
        """
    )

    ap.add_argument(
        "--data-path",
        help="""
        Path to KITTI dataset root (e.g. kitti/object_detection). Should contain a
        `training` and `testing` directory, each with a `image_2` directory containing
        PNG images. The `training/label_2` directory should contain a bunch of TXT
        labels.
        """,
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
        data_path / "training" / "image_2"
    ).is_dir(), "Given data path doesn't contain a subdirectory `training/image_2`"
    assert (
        data_path / "training" / "label_2"
    ).is_dir(), "Given data path doesn't contain a subdirectory `training/label_2`"
    assert (
        data_path / "testing"
    ).is_dir(), "Given data path doesn't contain a subdirectory `testing`"

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
        output_dir=output_dir,
    )


def parse_labels(detections_path: Path) -> List[BoundingBox]:
    detections = []
    with open(detections_path) as f:
        f_csv = csv.reader(f, delimiter=" ")
        for row in f_csv:
            x1, y1, x2, y2 = map(float, row[4:8])
            label = row[0]
            if label == "DontCare":
                continue
            else:
                detections.append(
                    BoundingBox(label=label, left=x1, right=x2, top=y1, bottom=y2)
                )
    return detections


def box2d_to_yolo(box2d: BoundingBox):
    label = LABEL_MAP[box2d.label]
    x1 = box2d.left / IMG_WIDTH
    x2 = box2d.right / IMG_WIDTH
    y1 = box2d.top / IMG_HEIGHT
    y2 = box2d.bottom / IMG_HEIGHT

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    width = abs(x2 - x1)
    height = abs(y2 - y1)

    return label, cx, cy, width, height


def convert_labels(labels_dir: Path, output_dir: Path):
    """
    This function converts KITTI labels (txt files) to YOLO compatible labels.

    The darknet format is
        <object id> <x> <y> <width> <height>
    """
    assert labels_dir.is_dir(), "Labels path is not a directory"
    assert output_dir.is_dir(), "Output directory doesn't exist"

    for label_txt in labels_dir.iterdir():
        if label_txt.suffix != ".txt":
            continue
        frame_file = output_dir / label_txt.name
        # Creates, opens, and adds to a txt file with the name of each label.txt
        with open(frame_file, "w+") as f:
            # Get the BoundingBoxes for each label
            for bbox in parse_labels(label_txt):
                if bbox.left >= bbox.right or bbox.top >= bbox.bottom:
                    continue
                label, cx, cy, width, height = box2d_to_yolo(bbox)
                f.write("{} {} {} {} {}\n".format(label, cx, cy, width, height))


if __name__ == "__main__":
    args = validate_args(parse_args())

    # First, copy each data directory over to the output directory.
    for dir in ["training", "testing"]:
        src = args.data_path / dir / "image_2"
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
        if dir == "training":
            labels_src = args.data_path / dir / "label_2"
            print("-- Generating labels at that dir in parallel")
            convert_labels(labels_src, dst)

        proc.wait()
        print("-- Done copying")

        if dir == "training":
            print("-- Removing images without corresponding labels")
            rm_imgs_without_labels(dst)

    # Create names file
    names = [''] * len(LABEL_MAP)
    for label, num in LABEL_MAP.items():
        names[num] = label
    names_file = args.output_dir / "kitti.names"
    with open(names_file, "w+") as f:
        f.write("\n".join(names))

