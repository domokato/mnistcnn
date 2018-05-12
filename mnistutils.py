import numpy as np


DATASETS_DIR = "/datasets/"
IMAGE_WIDTH = 28
TRAIN_X_COUNT = 50000
DEV_X_COUNT = 10000
TEST_X_COUNT = 10000
CHANNEL_COUNT = 1
CLASSES_COUNT = 10
BLACK = 255


def load():
    IMAGES_FILE_HEADER_SIZE = 16
    LABELS_FILE_HEADER_SIZE = 8

    train_x = np.fromfile(DATASETS_DIR + "train-images.idx3-ubyte", dtype=np.uint8)
    train_x = train_x[IMAGES_FILE_HEADER_SIZE:]
    train_x = train_x.reshape((TRAIN_X_COUNT+DEV_X_COUNT, IMAGE_WIDTH, IMAGE_WIDTH, CHANNEL_COUNT)).astype(float)
    dev_x = train_x[:DEV_X_COUNT]
    train_x = train_x[DEV_X_COUNT:]

    train_y = np.fromfile(DATASETS_DIR + "train-labels.idx1-ubyte", dtype=np.uint8)[LABELS_FILE_HEADER_SIZE:]
    dev_y = train_y[:DEV_X_COUNT]
    train_y = train_y[DEV_X_COUNT:]

    test_x = np.fromfile(DATASETS_DIR + "t10k-images.idx3-ubyte", dtype=np.uint8)
    test_x = test_x[IMAGES_FILE_HEADER_SIZE:]
    test_x = test_x.reshape((TEST_X_COUNT, IMAGE_WIDTH, IMAGE_WIDTH, CHANNEL_COUNT)).astype(float)

    test_y = np.fromfile(DATASETS_DIR + "t10k-labels.idx1-ubyte", dtype=np.uint8)[LABELS_FILE_HEADER_SIZE:]

    return train_x, train_y, dev_x, dev_y, test_x, test_y


def normalize(xs):
    return xs/BLACK
