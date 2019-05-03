from abc import ABC


class Selector(ABC):
    """
    A Selector object consists of a client used to select random images of same shape along with a metadata dict.
    """
    def poll_random_img(self):
        """
        Return a random image with replacement from the selector.

        :return: numpy array and metadata dict.
        """
        raise NotImplementedError()
