"""
Utility script to preprocess raw plus images into 28 by 10.
"""

import numpy as np
import os
from PIL import Image
from skimage.transform import resize

# Height and width in pixels. Both should be even.
HEIGHT = 28
WIDTH = 10

BASE_DIR = os.getcwd()
IMG_DIR = "{}/data/operators/plus/raw".format(BASE_DIR)
OUTPUT_PATH = "{}/data/operators/plus/post_processed.npy".format(BASE_DIR)


def load_jpg(img_path):
    """
    Load raw jpeg image into a 2D numpy array of values ranging from [0, 255].
    :param img_path: Path to jpg image.
    :return: numpy array containing image.
    """
    loaded_img = Image.open(img_path).convert('L')

    # Loads image as a 1D array.
    image_contents = np.array(loaded_img.getdata(), dtype=np.uint8)

    # Resizes to height by width.
    image_arr = np.resize(image_contents, (loaded_img.size[1], loaded_img.size[0]))

    return image_arr


def convert_jpgs(img_dir_path):
    """
    Loads all jpeg images in img_dir_path and converts them to a numpy array of size
    [N, H, W], where N is the number of images in the directory.

    The conversion is as follows:
    1. Resize the image to W by W.
    2. Get the average pixel value at the four corners in the image, and amplify the image to
       H by W.

    :param img_dir_path: directory containing the raw images.
    :return: numpy array containing all images.
    """
    images = []

    for fname in os.listdir(img_dir_path):
        img_path = "{}/{}".format(img_dir_path, fname)

        raw_img = load_jpg(img_path)

        resized_img = resize(raw_img, (WIDTH, WIDTH), preserve_range=True)

        # Take four corner values and average for background color.
        avg_four_corners = (resized_img[0][0] + resized_img[0][WIDTH - 1] + resized_img[WIDTH - 1][0]
                            + resized_img[WIDTH - 1][WIDTH - 1]) // 4

        # Assumes both HEIGHT and WIDTH are even.
        leftover = (HEIGHT - WIDTH) // 2

        # background is leftover by WIDTH with value avg_four_corners.
        background = np.full((leftover, WIDTH), avg_four_corners)

        final_img = np.concatenate([background, resized_img, background], axis=0)

        print("Finished processing image {}".format(fname))
        images.append(final_img)

    return np.array(images)


def write_transformed_images(img_dir_path, output_path):
    """
    Transforms all images in img_dir_path and writes to an output file.

    :param img_dir_path: directory containing the raw images.
    :param output_path: file path to write the resulting transformed images to.
    """
    transformed_images = convert_jpgs(img_dir_path)

    print("Writing images.")
    np.save(output_path, transformed_images)


if __name__ == "__main__":
    write_transformed_images(IMG_DIR, OUTPUT_PATH)
