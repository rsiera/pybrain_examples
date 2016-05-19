import pygame

from pybrain_examples.nn_pygame_pong.constants import resolution, width, height
from pybrain_examples.nn_pygame_pong.models import Player, Ball
from pybrain_examples.nn_pygame_pong.utils import check_collision, calc_input_parameters, calc_desired_height, setup_network, \
    normalize_speed, PygameController, display_object, stop_on_boundry

pygame.init()
screen = pygame.display.set_mode(resolution)

clock = pygame.time.Clock()


def main():
    net, data, trainer = setup_network()
    player1 = Player((0, 0))
    player2 = Player(((width - 10), 100))
    players = [player1, player2]
    ball = Ball((width / 2, height / 2))

    while True:
        screen.fill((0, 0, 0))
        clock.tick(150)

        for i in players:
            display_object(i, screen)
        display_object(ball, screen)
        input_position = calc_input_parameters(player2, ball)

        desired_y = calc_desired_height(player2, ball)
        data.addSample(input_position, desired_y)
        stop_on_boundry(player2)

        if ball.position.right > width - 1:
            player2.position.top = height / 2
            trainer.train()

        check_collision(players, ball)
        speed = net.activate(input_position)[0]
        normalized_speed = normalize_speed(speed)
        player2.speed[1] = normalized_speed

        PygameController.update_player(player1)
        pygame.display.update()


if __name__ == "__main__":
    main()
