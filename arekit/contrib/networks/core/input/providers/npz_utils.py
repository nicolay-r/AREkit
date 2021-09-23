import numpy as np


class NpzUtilsProvider(object):

    @staticmethod
    def save(data, target):
        np.savez(target, data)

    @staticmethod
    def load(source):
        npz_vocab_data = np.load(source)
        return npz_vocab_data['arr_0']
