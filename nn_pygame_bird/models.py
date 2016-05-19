from __future__ import unicode_literals

import pygame
from random import randint

from pygame.constants import SRCALPHA
from pygame.rect import Rect

from pybrain_examples.nn_pygame_bird.constants import width, height


class Bird(pygame.sprite.Sprite):
    WIDTH = HEIGHT = 80
    START_VELOCITY = .1
    STARTY = 200
    STARTX = 100

    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.x, self.y = Bird.STARTX, Bird.STARTY
        self.image = pygame.image.load("static/bird.png")
        self.image = pygame.transform.scale(self.image, (Bird.WIDTH, Bird.HEIGHT))
        self.mask = pygame.mask.from_surface(self.image)

    def update(self, y):
        self.y += y

    @property
    def rect(self):
        return Rect(self.x, self.y, Bird.WIDTH, Bird.HEIGHT)


class Pipe(pygame.sprite.Sprite):
    PIECE_WIDTH = 80
    PIECE_HEIGHT = 30
    ADD_RATE = 1750
    SCROLL_SPEED = .2

    def __init__(self):
        self.x = width - 1
        self.image = pygame.Surface((Pipe.PIECE_WIDTH, height), SRCALPHA)
        self.counted = False

        self.image.convert()
        self.image.fill((0, 0, 0, 0))
        total_pieces = int((height - 3 * Bird.HEIGHT - 3 * Pipe.PIECE_HEIGHT) / Pipe.PIECE_HEIGHT)
        self.bot_pieces = randint(1, total_pieces)
        self.top_pieces = total_pieces - self.bot_pieces

        pipe_img = pygame.image.load("static/pipe.png")
        pipe_img = pygame.transform.scale(pipe_img, (Pipe.PIECE_WIDTH, 80))

        for i in range(self.bot_pieces + 1):
            piece_pos = (0, height - i * Pipe.PIECE_HEIGHT)
            self.image.blit(pipe_img, piece_pos)

        for i in range(self.top_pieces + 1):
            self.image.blit(pipe_img, (0, i * Pipe.PIECE_HEIGHT))

        self.top_pieces += 1
        self.bot_pieces += 1
        self.bot_height = self.bot_pieces * Pipe.PIECE_HEIGHT

        self.mask = pygame.mask.from_surface(self.image)

    @property
    def rect(self):
        return Rect(self.x, 0, Pipe.PIECE_WIDTH, Pipe.PIECE_HEIGHT)

    @property
    def visible(self):
        return -Pipe.PIECE_WIDTH < self.x < width

    def update(self):
        self.x -= Pipe.SCROLL_SPEED

    def collides_with(self, bird):
        return pygame.sprite.collide_mask(self, bird)
