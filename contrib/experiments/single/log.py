import os
from arekit.contrib.experiments.experiment_io import BaseExperimentNeuralNetworkIO


def write_log(nn_io, log_names, log_values):
    assert(isinstance(nn_io, BaseExperimentNeuralNetworkIO))
    assert(isinstance(log_names, list))
    assert(isinstance(log_values, list))
    assert(len(log_names) == len(log_values))

    log_path = os.path.join(nn_io.get_model_root(), u"log.txt")

    with open(log_path, 'w') as f:
        for index, log_value in enumerate(log_values):
            f.write("{}: {}\n".format(log_names[index], log_value))
