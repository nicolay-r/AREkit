import numpy as np


class NpzRepositoryUtils(object):

    @staticmethod
    def save(data, target):
        np.savez(target, data)

    @staticmethod
    def load(source):
        data = np.load(source)
        return data['arr_0']
