from __future__ import unicode_literals

import os
import random
from decimal import Decimal, ROUND_HALF_UP

import cv2
import numpy as np
from pybrain import SoftmaxLayer
from pybrain.datasets import ClassificationDataSet
from pybrain.supervised import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.utilities import percentError

from pybrain_examples.nn_numbers_recognizer.constants import MEDIA_DIR, NUMBERS_OF_EXAMPLE

iscolor = cv2.CV_LOAD_IMAGE_GRAYSCALE
NUMBERS_TO_CLASS = {
    '001': 0, '002': 1, '003': 2, '004': 3, '005': 4, '006': 5, '007': 6, '008': 7, '009': 8, '010': 9
}


def flatten_img(img):
    flatten_img = []
    height, width = img.shape
    for y in xrange(0, height):
        pixels = list(img[y])
        normalized_pixels = [p / 255. for p in pixels]
        flatten_img.extend(normalized_pixels)
    return flatten_img


def setup_network(training_dataset, hidden_neurons):
    fnn = buildNetwork(
        training_dataset.indim, hidden_neurons, training_dataset.outdim, outclass=SoftmaxLayer)
    trainer = BackpropTrainer(fnn, dataset=training_dataset, learningrate=0.1, momentum=0.)
    return fnn, trainer


def create_dataset(files_path):
    dataset = ClassificationDataSet(40 * 30, 1, class_labels=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
    for filename in os.listdir(files_path):
        name, extension = filename.split('.')
        number = name.split('-')[0].replace('img', '')
        img_path = os.path.join(files_path, filename)
        cv2_img = cv2.imread(img_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)
        flattened_img = flatten_img(cv2_img)
        dataset.addSample(flattened_img, [NUMBERS_TO_CLASS[number]])

    dataset.assignClasses()
    dataset._convertToOneOfMany()
    return dataset


def print_statistics(dataset):
    print 'target', dataset.getField('target').transpose()[0]
    print 'training', dataset.calculateStatistics()


def get_percent_error(trainer, dataset):
    out = trainer.testOnClassData(dataset=dataset, verbose=True)
    print out
    error = percentError(out, dataset['class'])
    return error


def round_half_up(value):
    return Decimal(value).quantize(Decimal('1.'), rounding=ROUND_HALF_UP)


def prepare_input_img(test_files):
    input_img = os.path.join(MEDIA_DIR, 'input.png')

    files = os.listdir(test_files)
    first_filename = files[0]
    first_img = cv2.imread(os.path.join(test_files, first_filename))
    cv2.imwrite(input_img, first_img)

    for filename in files[1:NUMBERS_OF_EXAMPLE]:
        img_second = cv2.imread(os.path.join(test_files, filename))
        first_img = cv2.imread(input_img)
        vis = np.concatenate((first_img, img_second), axis=1)
        cv2.imwrite(input_img, vis)


def resize(img):
    return cv2.resize(img, (40, 30))


def create_noise(img, pixels=5):
    height, width = img.shape
    for dummy in xrange(pixels):
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)
        gray = random.randint(0, 255)
        img.itemset((y, x), gray)
    return img


def _ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def resize_files(directory, to_directory):
    _ensure_dir(to_directory)
    for root, dirs, filenames in os.walk(directory):
        for filename in filenames:
            cv2_img = cv2.imread(os.path.join(root, filename), iscolor)
            resized_30_40 = resize(cv2_img)
            cv2.imwrite(os.path.join(to_directory, filename), resized_30_40)


def noise_files(directory, to_directory, iterations=3):
    _ensure_dir(to_directory)
    for root, dirs, filenames in os.walk(directory):
        for f in filenames:
            name, extension = f.split('.')
            number, others = name.split('-')
            for _ in xrange(iterations):
                cv2_img = cv2.imread(os.path.join(root, f), iscolor)
                noised = create_noise(cv2_img, pixels=45)
                new_name = '%s-%s-%s.png' % (number, others, _)
                cv2.imwrite(os.path.join(to_directory, new_name), noised)
