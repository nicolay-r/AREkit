import numpy as np


class VocabRepositoryUtils(object):

    @staticmethod
    def save(data, target):
        np.savetxt(target, data, fmt='%s')

    @staticmethod
    def load(source):
        return np.loadtxt(source, dtype=str)
