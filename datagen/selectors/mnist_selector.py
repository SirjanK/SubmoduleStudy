import numpy as np
from mnist import MNIST

from datagen.selectors.selector import Selector


class MNISTSelector(Selector):
    """
    Client to select uniformly at random a mnist example along with its label.
    """

    LABEL_KEY = "label"
    MNIST_HEIGHT = 28
    MNIST_WIDTH = 28

    def __init__(self, data_path, is_training=True):
        """
        Initializes the mnist selector.

        :param data_path: Path to mnist dataset.
        :param is_training: mode for training vs testing.
        """
        mnist_client = MNIST(path=data_path, return_type='numpy')

        # To read in .gz files
        mnist_client.gz = True

        if is_training:
            self.images, self.labels = mnist_client.load_training()
        else:
            self.images, self.labels = mnist_client.load_testing()

        self.images = np.reshape(self.images, (self.images.shape[0], self.MNIST_HEIGHT, self.MNIST_WIDTH))

    def poll_random_img(self):
        random_index = np.random.choice(self.images.shape[0])
        return self.images[random_index], {
            self.LABEL_KEY: self.labels[random_index]
        }
