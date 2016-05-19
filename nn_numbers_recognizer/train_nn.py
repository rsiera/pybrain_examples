from __future__ import unicode_literals

from pybrain_examples.nn_numbers_recognizer.constants import RESIZED_TRAIN_DIR, RESIZED_VALIDATION_DIR, STATIC_DIR, \
    RESIZED_DIR
from pybrain_examples.nn_numbers_recognizer.utils import create_dataset, print_statistics, setup_network, \
    get_percent_error, resize_files, noise_files
from pybrain_examples.nn_numbers_recognizer.xml_writers import NeuralNetworkWriter

HIDDEN_NEURONS = 20

if __name__ == '__main__':
    resize_files(STATIC_DIR, RESIZED_DIR)
    noise_files(RESIZED_DIR, RESIZED_TRAIN_DIR, iterations=1)
    training_dataset = create_dataset(RESIZED_TRAIN_DIR)

    print_statistics(training_dataset)
    fnn, trainer = setup_network(training_dataset, HIDDEN_NEURONS)
    network_writer = NeuralNetworkWriter(trainer, HIDDEN_NEURONS)

    noise_files(RESIZED_DIR, RESIZED_VALIDATION_DIR, iterations=1)
    validation_dataset = create_dataset(RESIZED_VALIDATION_DIR)
    print_statistics(validation_dataset)

    for _ in range(100):
        trainer.trainEpochs(15)
        training_error = get_percent_error(trainer, training_dataset)
        validation_error = get_percent_error(trainer, validation_dataset)

        print "epoch: %4d  train error: %5.2f%%  validation error: %5.2f%%\n" % (
            trainer.totalepochs, training_error, validation_error)
        network_writer.write_to_file(fnn)

        # TODO: do it correctly
        if validation_error > 30 and training_error < 5:
            print 'Over-fitting'
            break
