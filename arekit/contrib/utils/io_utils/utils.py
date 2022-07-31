import collections
import logging
from os.path import join, exists

from arekit.common.experiment.data_type import DataType
from arekit.common.folding.base import BaseDataFolding
from arekit.contrib.utils.utils_folding import experiment_iter_index


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def join_dir_with_subfolder_name(subfolder_name, dir):
    """ Returns subfolder in in directory
    """
    assert(isinstance(subfolder_name, str))
    assert(isinstance(dir, str))

    target_dir = join(dir, "{}/".format(subfolder_name))
    return target_dir


def filename_template(data_type, data_folding):
    assert(isinstance(data_type, DataType))
    assert(isinstance(data_folding, BaseDataFolding))
    return "{data_type}-{iter_index}".format(data_type=data_type.name.lower(),
                                             iter_index=experiment_iter_index(data_folding))


def check_targets_existence(targets):
    assert (isinstance(targets, collections.Iterable))

    result = True
    for filepath in targets:
        assert(isinstance(filepath, str))

        existed = exists(filepath)
        logger.info("Check existence [{is_existed}]: {fp}".format(is_existed=existed, fp=filepath))
        if not existed:
            result = False

    return result
