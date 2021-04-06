from pathlib import Path

def rm_imgs_without_labels(data_dir: Path):
    """
    This function removes the images from the train folder if the correspining labels
    are not found in the .txt file.

    NOTE - Make sure you perform the conversion from the label to txt.
    The code performs the following function,

    - Takes the input dataset folder path, searches if the images with label information
      are present.
    - If not found, removes the image.

    :params
        image_path  - The directory where the training images are present
        label_path  - The directory where .txt file correspinding to each image is
                      present.
    """

    for file in data_dir.iterdir():
        if file.suffix == ".jpg":
            image = file
            # Corresponding label file name
            label = image.with_suffix(".txt")
            if not label.is_file():
                image.unlink()

