import numpy as np

from datagen.combiners.combiner import Combiner


class HorizontalCombiner(Combiner):
    """
    The HorizontalCombiner is an implementation of the Combiner interface such that
    variable types and lengths of selectors can be combined by horizontal concatenation.

    This requires output images from the selectors to all have identical heights.
    """

    def combine_random_selector_images(self):
        """
        Horizontally concatenates images from individual selectors.

        :return: numpy array along with metadata from each individual selector. The metadata dict contains
        a nested dictionary for each individual metadata of the selectors.
        """
        random_images = []
        combined_metadata = dict()
        image_no = 0

        for selector in self.selectors:
            image, metadata = selector.poll_random_img()

            combined_metadata[image_no] = metadata
            random_images.append(image)

            image_no += 1

        # Throws exception if heights of the images do not match.
        concatenated_image = np.concatenate(random_images, axis=1)

        return concatenated_image, combined_metadata
