from abc import ABC

from datagen.selectors.selector import Selector


class Combiner(Selector, ABC):
    """
    A Combiner object is a Selector that combines different Selectors together into a single image.
    """

    def __init__(self, selectors):
        """
        Initializes the combiner.

        :param selectors: List of selectors.
        """
        self.selectors = selectors

    def poll_random_img(self):
        image, metadata = self.combine_random_selector_images()

        return image, metadata

    def combine_random_selector_images(self):
        """
        Combine random selector images into one single image.

        :return: 2D numpy array and metadata dict.
        """
        raise NotImplementedError()
