import deepdish
import os

from datagen.selectors.mnist_selector import MNISTSelector
from datagen.selectors.mnist_digit_selector import MNISTDigitSelector
from datagen.selectors.plus_selector import PlusSelector
from datagen.combiners.horizontal_combiner import HorizontalCombiner


BASE_DATA_DIR = os.getcwd()
DATA_DIR = "{}/data/e2e".format(BASE_DATA_DIR)
MNIST_DATA_DIR = "{}/data/mnist".format(BASE_DATA_DIR)
PLUS_DATA_PATH = "{}/data/operators/plus/post_processed.npy".format(BASE_DATA_DIR)

TRAIN = 'train'
TEST = 'test'

NUM_TRAIN_POINTS = 1000000
NUM_TEST_POINTS = 100000

LOGGING_NUM_STEPS = 10000
BATCH_WRITE_SIZE = 50000

# Initialize selectors.
mnist_selector_train = MNISTSelector(MNIST_DATA_DIR, is_training=True)
mnist_selector_test = MNISTSelector(MNIST_DATA_DIR, is_training=False)
plus_selector = PlusSelector(PLUS_DATA_PATH)

mnist_digit_selectors_train = []
mnist_digit_selectors_test = []
for digit in range(10):
    mnist_digit_selectors_train.append(MNISTDigitSelector(digit, mnist_selector_train))
    mnist_digit_selectors_test.append(MNISTDigitSelector(digit, mnist_selector_test))

print("Finished initializing all base selectors.")


def generate_two_digit_sum_images(num_points, is_training):
    """
    Generator that generates num_points (image, label) pairs where the image pertains to
    a two digit sum, and label corresponds to the actual sum as an int.

    :param num_points: Number of data points to generate.
    :param is_training: Boolean flag to indicate whether a train set or test set is being generated.
    """
    # Set up a selector that corresponds to two digit sum.
    mnist_selector = mnist_selector_train if is_training else mnist_selector_test

    two_digit_sum_selector = HorizontalCombiner(
        HorizontalCombiner(
            mnist_selector,
            plus_selector
        ),
        mnist_selector
    )

    for _ in range(num_points):
        image, metadata = two_digit_sum_selector.poll_random_img()

        first_addend_val = metadata[HorizontalCombiner.LEFT_KEY][HorizontalCombiner.LEFT_KEY][MNISTSelector.LABEL_KEY]
        second_addend_val = metadata[HorizontalCombiner.RIGHT_KEY][MNISTSelector.LABEL_KEY]

        yield image, first_addend_val + second_addend_val


def generate_two_digit_sum_dataset(num_points, is_training):
    """
    Generator that yields (image, sum_val, output_image) datapoints. Here, image is the image of the two digit sum,
    sum_val is the numerical integer value of the sum, and output_image is a randomly sampled image of the output value.

    :param num_points: Number of data points to generate.
    :param is_training: Boolean flag to indicate whether a train set or test set is being generated.
    """
    for image, sum_val in generate_two_digit_sum_images(num_points, is_training):
        tens_digit, ones_digit = sum_val // 10, sum_val % 10
        mnist_selector_tens = mnist_digit_selectors_train[tens_digit] if is_training \
            else mnist_digit_selectors_test[tens_digit]
        mnist_selector_ones = mnist_digit_selectors_train[ones_digit] if is_training \
            else mnist_digit_selectors_test[ones_digit]

        sum_selector = HorizontalCombiner(mnist_selector_tens, mnist_selector_ones)
        sum_img, _ = sum_selector.poll_random_img()

        yield image, sum_val, sum_img


def write_data(output_dir, data_generator):
    """
    Write data from the data_generator to output_path. The data_generator returns (image, sum_val, output_image)
    data points.
    :param output_dir: Path to output the data.
    :param data_generator: Finite generator to output data points.
    """
    images = []
    sums = []
    output_images = []

    processed_image_cnt = 0
    batch_index = 0

    # Iterate through all data points.
    for image, sum_val, output_image in data_generator:
        images.append(image)
        sums.append(sum_val)
        output_images.append(output_image)

        processed_image_cnt += 1
        if processed_image_cnt % LOGGING_NUM_STEPS == 0:
            print("Processed {} images".format(processed_image_cnt))

        # Once we have processed enough to constitute one batch, we flush to disk to free up memory.
        if processed_image_cnt % BATCH_WRITE_SIZE == 0:
            print("Writing batch: {}".format(batch_index))

            batch_data = {
                'images': images,
                'sum_values': sums,
                'output_images': output_images
            }

            output_path = "{}/part-{}.h5".format(output_dir, batch_index)
            deepdish.io.save(output_path, batch_data)

            print("Finished writing batch.")

            # Empty out the lists for new batch.
            images = []
            sums = []
            output_images = []

            batch_index += 1


def write_two_digit_sum_dataset(output_train_path, output_test_path, num_train, num_test):
    """
    Writes the dataset of the two digit sum.
    :param output_train_path: Path to output train data.
    :param output_test_path: Path to output test data.
    :param num_train: Number of training points.
    :param num_test: Number of testing points.
    """
    training_data_generator = generate_two_digit_sum_dataset(num_train, is_training=True)
    test_data_generator = generate_two_digit_sum_dataset(num_test, is_training=False)

    print("Starting to write training data.")
    write_data(output_train_path, training_data_generator)
    print("Finished writing training data.")

    print("Starting to write test data.")
    write_data(output_test_path, test_data_generator)
    print("Finished writing test data.")


if __name__ == '__main__':
    write_two_digit_sum_dataset(
        output_train_path="{}/{}".format(DATA_DIR, TRAIN),
        output_test_path="{}/{}".format(DATA_DIR, TEST),
        num_train=NUM_TRAIN_POINTS,
        num_test=NUM_TEST_POINTS
    )
