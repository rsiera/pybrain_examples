from __future__ import unicode_literals

import os

BASE_DIR = os.path.dirname(__file__)
STATIC_DIR = os.path.join(BASE_DIR, 'static')
MEDIA_DIR = os.path.join(BASE_DIR, 'media')
NETWORKS_DIR = os.path.join(BASE_DIR, 'networks')

RESIZED_DIR = os.path.join(MEDIA_DIR, 'resized_40_30')
RESIZED_DAMAGED_TEST_DIR = os.path.join(MEDIA_DIR, 'resized_40_30_damaged')
RESIZED_TRAIN_DIR = os.path.join(MEDIA_DIR, 'resized_40_30_train')
RESIZED_VALIDATION_DIR = os.path.join(MEDIA_DIR, 'resized_40_30_validation')

NN_OUTPUT_TO_NUMBERS = {
    tuple([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]): '0',
    tuple([0, 1, 0, 0, 0, 0, 0, 0, 0, 0]): '1',
    tuple([0, 0, 1, 0, 0, 0, 0, 0, 0, 0]): '2',
    tuple([0, 0, 0, 1, 0, 0, 0, 0, 0, 0]): '3',
    tuple([0, 0, 0, 0, 1, 0, 0, 0, 0, 0]): '4',
    tuple([0, 0, 0, 0, 0, 1, 0, 0, 0, 0]): '5',
    tuple([0, 0, 0, 0, 0, 0, 1, 0, 0, 0]): '6',
    tuple([0, 0, 0, 0, 0, 0, 0, 1, 0, 0]): '7',
    tuple([0, 0, 0, 0, 0, 0, 0, 0, 1, 0]): '8',
    tuple([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]): '9',
}

NUMBERS_OF_EXAMPLE = 100
