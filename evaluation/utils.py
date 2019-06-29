import collections
from core.evaluation.cmp_opinions import OpinionCollectionsToCompare


class OpinionCollectionsToCompareUtils:

    def __init__(self):
        pass

    @staticmethod
    def iter_comparable_collections(test_filepath_func,
                                    etalon_filepath_func,
                                    read_collection_func,
                                    indices):
        assert(isinstance(indices, collections.Iterable))

        for index in indices:
            yield OpinionCollectionsToCompare(test_filepath_func=test_filepath_func,
                                              etalon_filepath_func=etalon_filepath_func,
                                              read_collection_func=read_collection_func,
                                              index=index)



