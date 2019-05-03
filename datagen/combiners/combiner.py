from abc import ABC

from datagen.selectors.selector import Selector


class Combiner(Selector, ABC):
    """
    A Combiner object is a Selector that combines two different Selectors together into a single image.
    """

    def __init__(self, selector1, selector2):
        """
        Initializes the combiner.

        :param selector1: first selector the combiner operates on.
        :param selector2: second selector the combiner operates on.
        """
        self.selector1 = selector1
        self.selector2 = selector2

    def poll_random_img(self):
        image, metadata = self.combine_random_selector_images()

        return image, metadata

    def combine_random_selector_images(self):
        """
        Combine two random selector images into one single image.

        :return: 2D numpy array and metadata dict.
        """
        raise NotImplementedError()
