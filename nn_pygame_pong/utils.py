from __future__ import unicode_literals

import pygame
import random

from pybrain.datasets import SupervisedDataSet
from pybrain.supervised import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork

from pybrain_examples.nn_pygame_pong.constants import height


def check_collision(players, ball):
    for i in players:
        if ball.position.colliderect(i.position):
            ball.speed[0] *= -1
            ball.speed[1] = random.choice((-1, 1))
            if ball.position.left < 200:
                ball.position.left += 5
            if ball.position.right > 200:
                ball.position.right -= 5


def calc_input_parameters(player2, newball):
    x = player2.position.center[0] - newball.position.center[0]
    y = newball.position.center[1] - player2.position.center[1]
    return [y, x]


def calc_desired_height(player2, newball):
    return newball.position.center[1] - player2.position.center[1]


def setup_network():
    net = buildNetwork(2, 3, 1)
    data = SupervisedDataSet(2, 1)
    trainer = BackpropTrainer(net, data)
    return net, data, trainer


def normalize_speed(p):
    if p > 3:
        p = 3
    if p < -3:
        p = -3
    return p


class PygameController(object):
    CONTROLLS_MAPPING = {
        pygame.KEYDOWN: {
            pygame.K_UP: -3,
            pygame.K_DOWN: 3
        },
        pygame.KEYUP: {
            pygame.K_UP: 0,
            pygame.K_DOWN: 0
        }
    }

    @classmethod
    def update_player(cls, player):
        speed = cls.get_events()
        if speed is not None:
            player.speed[1] = speed

    @classmethod
    def get_events(cls):
        for event in pygame.event.get():
            event_type_control = cls.CONTROLLS_MAPPING.get(event.type, {})
            if hasattr(event, 'key'):
                return event_type_control.get(event.key)


def display_object(object, screen):
    object.show(screen)
    object.move()


def stop_on_boundry(player2):
    if player2.position.top < 0:
        player2.position.top = 0
        player2.speed[1] = -player2.speed[1]
    if player2.position.bottom > height:
        player2.speed[1] = -player2.speed[1]
        player2.position.bottom = height
