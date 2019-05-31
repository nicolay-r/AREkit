class FilesToCompare:

    def __init__(self, test_filepath, etalon_filepath, index):
        assert(isinstance(test_filepath, unicode))
        assert(isinstance(etalon_filepath, unicode))
        assert(isinstance(index, int))
        self.__test_fp = test_filepath
        self.__etalon_fp = etalon_filepath
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
