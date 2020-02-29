from os.path import join
from arekit.common.utils import create_dir_if_not_exists
from arekit.contrib.experiments.io_utils_base import BaseExperimentsIOUtils


def get_path_of_subfolder_in_experiments_dir(subfolder_name, experiments_io):
    """
    Returns subfolder in experiments directory
    """
    assert(isinstance(subfolder_name, unicode))
    assert(isinstance(experiments_io, BaseExperimentsIOUtils))

    target_dir = join(experiments_io.get_experiments_dir(), u"{}/".format(subfolder_name))
    create_dir_if_not_exists(target_dir)
    return target_dir


