from __future__ import unicode_literals

from pybrain_examples.nn_prediction.utils import create_dataset, print_statistics, setup_network, get_percent_error
from pybrain_examples.nn_prediction.xml_writers import NeuralNetworkWriter


HIDDEN_NEURONS = 20

if __name__ == '__main__':
    training_dataset = create_dataset('2013-2014-PR.csv')
    print_statistics(training_dataset)
    fnn, trainer = setup_network(training_dataset, HIDDEN_NEURONS)

    validation_dataset = create_dataset('2013-2014-PR-validation.csv')
    print_statistics(validation_dataset)

    writer = NeuralNetworkWriter(trainer, HIDDEN_NEURONS)

    for _ in range(100):
        trainer.trainEpochs(50)
        training_error = get_percent_error(trainer, training_dataset)
        validation_error = get_percent_error(trainer, validation_dataset)
        print "epoch: %4d  train error: %5.2f%%  validation error: %5.2f%%\n" % (
            trainer.totalepochs, training_error, validation_error)
        writer.write_to_file(fnn)
