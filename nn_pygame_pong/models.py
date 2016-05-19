from __future__ import unicode_literals

import pygame
import random

from pybrain_examples.nn_pygame_pong.constants import INITIAL_SPEED, width, height, PLAYER_COLOR, PLAYER_RECT, BALL_RECT, \
    BALL_COLOR


class Player(object):
    def __init__(self, start_position):
        self.image = pygame.Surface((10, 40))
        self.position = self.image.get_rect()
        self.position.topleft = start_position
        pygame.draw.rect(self.image, PLAYER_COLOR, PLAYER_RECT, 0)
        self.speed = list(INITIAL_SPEED)

    def move(self):
        self.position = self.position.move(self.speed)

    def show(self, screen):
        screen.blit(self.image, self.position)


class Ball(object):
    def __init__(self, start_position):
        self.image = pygame.Surface((10, 10))
        self.position = self.image.get_rect()
        self.position.topleft = start_position
        pygame.draw.circle(self.image, BALL_COLOR, BALL_RECT, 5)
        self.speed = list(INITIAL_SPEED)
        self.restartball()

    def show(self, screen):
        screen.blit(self.image, self.position)

    def move(self):
        self.position = self.position.move(self.speed)
        if self.position.right > width:
            self.restartball()
        if self.position.bottom > height:
            self.speed[1] *= -1
        if self.position.top < 0:
            self.speed[1] *= -1
        if self.position.left < 0:
            self.restartball()

    def restartball(self):
        self.position.topleft = ((width / 2 + 30), (height / 2 + 25))
        self.speed = [random.choice((1, 2)), random.choice((-4, -3, -2, -1, 1, 2, 3, 4))]
