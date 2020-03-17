import os
from os import path
from os.path import dirname, join


def get_data_root():
    return path.join(dirname(__file__), u"data/")


def get_experiments_dir():
    target_dir = join(get_data_root(), u"./rusentrel/")
    create_dir_if_not_exists(target_dir)
    return target_dir


def create_dir_if_not_exists(filepath):
    dir = os.path.dirname(filepath)
    if not os.path.exists(dir):
        os.makedirs(dir)


def get_path_of_subfolder_in_experiments_dir(subfolder_name):
    """
    Returns subfolder in experiments directory
    """
    assert(isinstance(subfolder_name, unicode))
    target_dir = os.path.join(get_experiments_dir(), u"{}/".format(subfolder_name))
    create_dir_if_not_exists(target_dir)
    return target_dir

