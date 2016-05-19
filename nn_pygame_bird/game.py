import pygame
from collections import deque

from pybrain_examples.nn_pygame_bird.constants import width, height
from pybrain_examples.nn_pygame_bird.models import Bird, Pipe
from pybrain_examples.nn_pygame_bird.utils import calc_desired_height, outside_boundary, calc_input_parameters

pygame.init()
screen = pygame.display.set_mode((width, height))


def train(ai, data, trainer, network_writer, headless=False):
    bird = Bird()
    pipes = deque()
    frame = 0

    while True:
        bird.update(0)
        if not headless:
            screen.fill((0, 255, 255))
            screen.blit(bird.image, (bird.x, bird.y))

        if frame % Pipe.ADD_RATE == 0:
            p = Pipe()
            pipes.append(p)

        while pipes and not pipes[0].visible:
            pipes.popleft()

        for p in pipes:
            if p.x + Pipe.PIECE_WIDTH / 2 < bird.x and not p.counted:
                p.counted = True

            p.update()
            if not headless:
                screen.blit(p.image, (p.x, 0))

        input_parameters = calc_input_parameters(bird, pipes)
        desired_y = calc_desired_height(bird, pipes[0])
        data.addSample(input_parameters, desired_y)

        if frame % 100 == 0:
            trainer.train()

        output = ai.activate(input_parameters)
        direction = -0.1 if output > 0.5 else 0.1
        bird.update(direction)

        if not headless:
            pygame.display.flip()

        frame += 1
        if frame % 1000 == 0:
            network_writer.write_to_file(ai)


def play(ai, headless=False):
    bird = Bird()
    pipes = deque()
    frame = 0

    while True:
        bird.update(0)
        if not headless:
            screen.fill((0, 255, 255))
            screen.blit(bird.image, (bird.x, bird.y))

        if frame % Pipe.ADD_RATE == 0:
            p = Pipe()
            pipes.append(p)

        col = any(p.collides_with(bird) for p in pipes)
        if outside_boundary(bird, col):
            break

        while pipes and not pipes[0].visible:
            pipes.popleft()

        for p in pipes:
            if p.x + Pipe.PIECE_WIDTH / 2 < bird.x and not p.counted:
                p.counted = True

            p.update()
            if not headless:
                screen.blit(p.image, (p.x, 0))

        input_parameters = calc_input_parameters(bird, pipes)
        output = ai.activate(input_parameters)
        direction = -0.1 if output > 0.5 else 0.1
        bird.update(direction)

        if not headless:
            pygame.display.flip()
        frame += 1

