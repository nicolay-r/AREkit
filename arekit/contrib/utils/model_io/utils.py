from os.path import join

from arekit.common.folding.base import BaseDataFolding
from arekit.contrib.utils.cv.two_class import TwoClassCVFolding


def join_dir_with_subfolder_name(subfolder_name, dir):
    """ Returns subfolder in in directory
    """
    assert(isinstance(subfolder_name, str))
    assert(isinstance(dir, str))

    target_dir = join(dir, "{}/".format(subfolder_name))
    return target_dir


def experiment_iter_index(folding):
    assert(isinstance(folding, BaseDataFolding))

    if isinstance(folding, TwoClassCVFolding):
        return folding.StateIndex

    # In other cases we consider that there is only a single state.
    return 0
