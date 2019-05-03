import numpy as np

from datagen.combiners.combiner import Combiner


class VerticalCombiner(Combiner):
    """
    The VerticalCombiner is an implementation of the Combiner interface such that
    variable types of two selectors can be combined by vertical concatenation.

    This requires output images from the selectors to all have identical widths.
    The first selector passed in will be the top image (top-down approach).
    """

    TOP_KEY = "top"
    BOTTOM_KEY = "bottom"

    def combine_random_selector_images(self):
        """
        Vertically concatenates images from individual selectors.

        :return: numpy array along with metadata from each individual selector. The metadata dict contains
        a nested dictionary for each individual metadata of the selectors keyed by "top" and "bottom".
        """
        top_img, top_metadata = self.selector1.poll_random_img()
        bottom_img, bottom_metadata = self.selector2.poll_random_img()

        combined_metadata = {
            self.TOP_KEY: top_metadata,
            self.BOTTOM_KEY: bottom_metadata
        }

        # Throws exception if widths of the images do not match.
        concatenated_image = np.concatenate([top_img, bottom_img], axis=0)

        return concatenated_image, combined_metadata
