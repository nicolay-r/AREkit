class OpinionCollectionsToCompare(object):

    def __init__(self,
                 test_filepath_func,
                 etalon_filepath_func,
                 read_collection_func,
                 index):
        assert(callable(test_filepath_func))
        assert(callable(etalon_filepath_func))
        assert(callable(read_collection_func))
        assert(isinstance(index, int))

        self.__test_opinions = read_collection_func(test_filepath_func(index))
        self.__etalon_opinions = read_collection_func(etalon_filepath_func(index))
        self.__index = index

    @property
    def TestOpinionCollection(self):
        return self.__test_opinions

    @property
    def EtalonOpinionCollection(self):
        return self.__etalon_opinions

    @property
    def index(self):
        return self.__index