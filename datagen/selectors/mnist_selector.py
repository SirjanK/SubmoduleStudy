import numpy as np
from mnist import MNIST

from datagen.selectors.selector import Selector


class MNISTSelector(Selector):
    """
    Client to select uniformly at random a mnist example along with its label.
    """

    def __init__(self, data_path, is_training=True):
        """
        Initializes the mnist selector.

        :param data_path: Path to mnist dataset.
        :param is_training: mode for training vs testing.
        """
        mnist_client = MNIST(path=data_path, return_type='numpy')

        if is_training:
            self.images, self.labels = mnist_client.load_training()
        else:
            self.images, self.labels = mnist_client.load_testing()

    def poll_random_img(self):
        random_index = np.random.choice(self.images.shape[0])
        return self.images[random_index], {
            "label": self.labels[random_index]
        }
