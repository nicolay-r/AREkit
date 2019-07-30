from os import path
from os.path import dirname


def get_data_root():
    return path.join(dirname(__file__), u"data/")
