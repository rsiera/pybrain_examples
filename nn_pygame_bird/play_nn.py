import os

from pybrain.tools.xml import NetworkReader

from game import play
from pybrain_examples.nn_pygame_bird.constants import NETWORKS_DIR

if __name__ == '__main__':
    filename = '2016_05_19/10_50.xml'
    net = NetworkReader.readFrom(os.path.join(NETWORKS_DIR, filename))
    play(net, False)
