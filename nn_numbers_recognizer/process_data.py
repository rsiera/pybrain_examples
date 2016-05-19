from __future__ import unicode_literals

import os

import cv2
import numpy as np
from pybrain.tools.xml import NetworkReader

from pybrain_examples.nn_numbers_recognizer.constants import NN_OUTPUT_TO_NUMBERS, NETWORKS_DIR, \
    RESIZED_DAMAGED_TEST_DIR, STATIC_DIR, RESIZED_DIR
from pybrain_examples.nn_numbers_recognizer.utils import flatten_img, round_half_up, prepare_input_img, resize_files, \
    noise_files

WINDOW_HEIGHT, WINDOW_WIDTH = 40, 30
FONT_COLOR = (0, 0, 0)


# Iterates over image and write in right top corner prediction about number
# filename - name of the NN
# media/input.png - input image
# media/answers.png - output image


def slide_window_mark_numbers(net):
    input_image = cv2.imread('media/input.png', cv2.CV_LOAD_IMAGE_GRAYSCALE)
    img_height, img_width = input_image.shape
    for y in xrange(0, img_height, WINDOW_HEIGHT):
        for x in xrange(0, img_width, WINDOW_WIDTH):
            roi = input_image[y:y + WINDOW_HEIGHT, x:x + WINDOW_WIDTH]
            input_data = flatten_img(roi)
            out = net.activate(input_data)
            result = tuple([int(round_half_up(o)) for o in out])
            number = NN_OUTPUT_TO_NUMBERS.get(result, '')
            first_roi = cv2.imread('media/answers.png', cv2.CV_LOAD_IMAGE_GRAYSCALE)
            cv2.putText(roi, number, (15, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, FONT_COLOR, 2)
            vis = np.concatenate((first_roi, roi), axis=1)
            cv2.imwrite('media/answers.png', vis)


def main():
    filename = '15_05/100_405.xml'
    resize_files(STATIC_DIR, RESIZED_DIR)
    noise_files(RESIZED_DIR, RESIZED_DAMAGED_TEST_DIR)
    prepare_input_img(RESIZED_DAMAGED_TEST_DIR)
    cv2.imwrite('media/answers.png', np.ones((40, 30, 1)))

    net = NetworkReader.readFrom(os.path.join(NETWORKS_DIR, filename))
    slide_window_mark_numbers(net)

if __name__ == "__main__":
    main()
