from os.path import join, exists

from arekit.common.experiment.formats.base import BaseExperiment
from arekit.common.experiment.io_utils import BaseIOUtils
from arekit.common.utils import create_dir_if_not_exists


def mark_dir_for_serialization(io_utils, logger, experiment, skip_if_folder_exists):
    assert(issubclass(io_utils, BaseIOUtils))
    assert(isinstance(experiment, BaseExperiment))
    assert(isinstance(skip_if_folder_exists, bool))

    target_dir = io_utils.get_target_dir(experiment)
    target_file = join(target_dir, 'lock.txt')
    if exists(target_file) and skip_if_folder_exists:
        logger.info("TARGET DIR EXISTS: {}".format(target_dir))
        return
    else:
        create_dir_if_not_exists(target_dir)
        open(target_file, 'a').close()