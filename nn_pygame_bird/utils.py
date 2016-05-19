from __future__ import unicode_literals

from pybrain_examples.nn_pygame_bird.constants import height


def calc_desired_height(bird, pipe):
    return 1 if bird.y > pipe.bot_height else 0


def outside_boundary(bird, col):
    return bird.y > 600 or bird.y < 0 or col


def calc_input_parameters(bird, pipes):
    return [bird.x - pipes[0].x, height - bird.y - pipes[0].bot_height]
