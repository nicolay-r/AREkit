import collections

from arekit.common.evaluation.pairs.func_based import FuncBasedDataPairsToCompare


class DataPairsIterators:
    """ Provides a variety ways of how the data might be iterated.
        (In most cases by a given set of document identifiers)
    """

    def __init__(self):
        pass

    @staticmethod
    def iter_func_based_collections(doc_ids,
                                    read_test_collection_func,
                                    read_etalon_collection_func):
        """ Funtion-Based data-pairs iterator
        """
        assert(isinstance(doc_ids, collections.Iterable))

        for doc_id in doc_ids:
            yield FuncBasedDataPairsToCompare(doc_id=doc_id,
                                              read_test_data_func=read_test_collection_func,
                                              read_etalon_data_func=read_etalon_collection_func)
