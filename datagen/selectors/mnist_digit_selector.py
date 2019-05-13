import numpy as np

from datagen.selectors.selector import Selector
from datagen.selectors.mnist_selector import MNISTSelector


class MNISTDigitSelector(Selector):
    """
    Client to select uniformly at random a mnist example corresponding to a specific digit.
    """

    def __init__(self, digit, mnist_selector):
        """
        Initializes the mnist selector.

        :param digit: Digit (0-9 inclusive integer) we want to retrieve images for.
        :param mnist_selector: An MNISTSelector used to only load MNIST data once into memory.
        """
        self.all_images = mnist_selector.images
        self.all_labels = mnist_selector.labels
        self.indices_with_digit = np.where(self.all_labels == digit)[0]

    def poll_random_img(self):
        random_index = np.random.choice(self.indices_with_digit)
        return self.all_images[random_index], {
            MNISTSelector.LABEL_KEY: self.all_labels[random_index]
        }
