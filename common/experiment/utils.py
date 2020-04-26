from os.path import join

from arekit.common.utils import create_dir_if_not_exists


def get_path_of_subfolder_in_experiments_dir(subfolder_name, experiments_dir):
    """
    Returns subfolder in experiments directory
    """
    assert(isinstance(subfolder_name, unicode))
    assert(isinstance(experiments_dir, unicode))

    target_dir = join(experiments_dir, u"{}/".format(subfolder_name))
    create_dir_if_not_exists(target_dir)
    return target_dir