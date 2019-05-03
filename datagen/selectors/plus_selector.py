import numpy as np

from datagen.selectors.selector import Selector


class PlusSelector(Selector):
    """
    Client to select uniformly at random an example of a "plus" image.
    """

    def __init__(self, data_path):
        """
        Initializes the plus selector.

        :param data_path: Path to the plus dataset path.
        """
        self.images = np.load(data_path)

    def poll_random_img(self):
        random_index = np.random.choice(self.images.shape[0])

        # Return an empty dict; no metadata required.
        return self.images[random_index], {}
