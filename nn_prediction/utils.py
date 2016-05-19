from __future__ import unicode_literals

from pybrain import SoftmaxLayer
from pybrain.datasets import ClassificationDataSet
from pybrain.supervised import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.utilities import percentError

from pybrain_examples.nn_prediction.csv_models import FootballDataCsv

MIN_NORMALIZED = 0
MAX_NORMALIZED = 1


def normalize(x, min_value, max_value):
    return ((x - min_value) / (max_value - min_value)) * (MAX_NORMALIZED - MIN_NORMALIZED) + MIN_NORMALIZED


def setup_network(training_dataset, hidden_neurons):
    fnn = buildNetwork(
        training_dataset.indim, hidden_neurons, training_dataset.outdim, outclass=SoftmaxLayer)
    trainer = BackpropTrainer(fnn, dataset=training_dataset, learningrate=0.1, momentum=0.)
    return fnn, trainer


def create_dataset(filename):
    dataset = ClassificationDataSet(13, 1, class_labels=['0', '1', '2'])
    football_data = FootballDataCsv(filename)
    total_min = football_data.total_min()
    total_max = football_data.total_max()
    for data in football_data:
        normalized_features = [normalize(x, min_value=total_min, max_value=total_max) for x in data.to_list()]
        dataset.addSample(normalized_features, [data.binarized_output])
    dataset.assignClasses()
    dataset._convertToOneOfMany()
    return dataset


def print_statistics(dataset):
    print 'target', dataset.getField('target').transpose()[0]
    print 'training', dataset.calculateStatistics()


def get_percent_error(trainer, dataset):
    out = trainer.testOnClassData(dataset=dataset, verbose=True)
    error = percentError(out, dataset['class'])
    return error
