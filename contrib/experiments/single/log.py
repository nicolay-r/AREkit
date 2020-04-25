import os
from arekit.contrib.experiments.data_io import DataIO


def write_log(data_io, log_names, log_values):
    assert(isinstance(data_io, DataIO))
    assert(isinstance(log_names, list))
    assert(isinstance(log_values, list))
    assert(len(log_names) == len(log_values))

    log_path = os.path.join(data_io.get_model_root(), u"log.txt")

    with open(log_path, 'w') as f:
        for index, log_value in enumerate(log_values):
            f.write("{}: {}\n".format(log_names[index], log_value))
