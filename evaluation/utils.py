import collections


class FilesToCompareUtils:

    def __init__(self):
        pass

    @staticmethod
    def get_list_of_comparable_files(test_filepath_func, etalon_filepath_func, indices):
        assert(isinstance(indices, collections.Iterable))

        return [FilesToCompare(test_filepath_func=test_filepath_func,
                               etalon_filepath_func=etalon_filepath_func,
                               index=index)
                for index in indices]


class FilesToCompare:

    def __init__(self, test_filepath_func, etalon_filepath_func, index):
        assert(callable(test_filepath_func))
        assert(callable(etalon_filepath_func))
        assert(isinstance(index, int))
        self.__test_fp = test_filepath_func(index)
        self.__etalon_fp = etalon_filepath_func(index)
        self.__index = index

    @property
    def TestFilepath(self):
        return self.__test_fp

    @property
    def EtalonFilepath(self):
        return self.__etalon_fp

    @property
    def index(self):
        return self.__index
