import os
from arekit.contrib.experiments.base import BaseExperiment


def write_log(experiment, log_names, log_values):
    assert(isinstance(experiment, BaseExperiment))
    assert(isinstance(log_names, list))
    assert(isinstance(log_values, list))
    assert(len(log_names) == len(log_values))

    log_path = os.path.join(experiment.get_model_root(), u"log.txt")

    with open(log_path, 'w') as f:
        for index, log_value in enumerate(log_values):
            f.write("{}: {}\n".format(log_names[index], log_value))
