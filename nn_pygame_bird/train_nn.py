from pybrain.datasets import SupervisedDataSet
from pybrain.supervised import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork

from game import train
from pybrain_examples.nn_pygame_bird.xml_writers import NeuralNetworkWriter

HIDDEN_NEURONS = 50


def setup_network():
    net = buildNetwork(2, HIDDEN_NEURONS, 1)
    data = SupervisedDataSet(2, 1)
    trainer = BackpropTrainer(net, data)
    return net, data, trainer


net, data, trainer = setup_network()
network_writer = NeuralNetworkWriter(trainer, HIDDEN_NEURONS)
train(net, data, trainer, network_writer, False)
