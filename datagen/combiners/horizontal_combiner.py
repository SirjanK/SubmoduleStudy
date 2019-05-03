import numpy as np

from datagen.combiners.combiner import Combiner


class HorizontalCombiner(Combiner):
    """
    The HorizontalCombiner is an implementation of the Combiner interface such that
    variable types of two selectors can be combined by horizontal concatenation.

    This requires output images from the selectors to all have identical heights.
    """

    LEFT_KEY = "left"
    RIGHT_KEY = "right"

    def combine_random_selector_images(self):
        """
        Horizontally concatenates images from individual selectors.

        :return: numpy array along with metadata from each individual selector. The metadata dict contains
        a nested dictionary for each individual metadata of the selectors keyed by "left" and "right".
        """
        left_image, left_metadata = self.selector1.poll_random_img()
        right_image, right_metadata = self.selector2.poll_random_img()

        combined_metadata = {
            self.LEFT_KEY: left_metadata,
            self.RIGHT_KEY: right_metadata
        }

        # Throws exception if heights of the images do not match.
        concatenated_image = np.concatenate([left_image, right_image], axis=1)

        return concatenated_image, combined_metadata
