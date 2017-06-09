import os
import re
import numpy as np
from skimage import io, transform, color


# str => [str]
def get_files(folder, limit=None):
    raw_files = os.listdir(os.path.join(folder))
    if limit:
        raw_files = raw_files[:limit]

    images = filter(lambda x: re.match(r'.*\.jpg$', x), raw_files)

    return map(lambda x: os.path.join(folder, x), images)


def get_batch(batch_size, file_list):
    batches = []
    i = 0
    while (i+1)*batch_size < len(file_list):
        batches.append(file_list[batch_size * i: batch_size * (i+1)])
        i += 1

    return batches


def rotate90(image):
    return transform.rotate(image, 90)


def rotate180(image):
    return transform.rotate(image, 180)


def rotate270(image):
    return transform.rotate(image, 270)


def flip(image):
    return np.fliplr(image)


def proc_batch(batch):
    batch_images = io.imread_collection(batch)
    return color.rgb2lab(batch_images)


def sample_files(np_file_list, sample_size):
    return np_file_list[np.random.choice(len(np_file_list), sample_size, replace=False)]

