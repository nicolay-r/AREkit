from collections.abc import Iterable
import logging
from os.path import join, exists

from arekit.common.experiment.data_type import DataType


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def join_dir_with_subfolder_name(subfolder_name, dir):
    """ Returns subfolder in in directory
    """
    assert(isinstance(subfolder_name, str))
    assert(isinstance(dir, str))

    target_dir = join(dir, "{}/".format(subfolder_name))
    return target_dir


def filename_template(data_type):
    assert(isinstance(data_type, DataType))
    return "{data_type}-0".format(data_type=data_type.name.lower())


def check_targets_existence(targets):
    assert (isinstance(targets, Iterable))

    result = True
    for filepath in targets:
        assert(isinstance(filepath, str))

        existed = exists(filepath)
        logger.info("Check existence [{is_existed}]: {fp}".format(is_existed=existed, fp=filepath))
        if not existed:
            result = False

    return result
