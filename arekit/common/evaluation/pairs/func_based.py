from arekit.common.evaluation.pairs.base import BasePairToCompare


class FuncBasedDataPairsToCompare(BasePairToCompare):
    """ Function-based data provider.
    """

    def __init__(self, doc_id, read_etalon_data_func, read_test_data_func):
        assert(callable(read_etalon_data_func))
        assert(callable(read_test_data_func))

        super(FuncBasedDataPairsToCompare, self).__init__(doc_id)

        self.__test_data = read_test_data_func(doc_id)
        self.__etalon_data = read_etalon_data_func(doc_id)

    @property
    def TestData(self):
        return self.__test_data

    @property
    def EtalonData(self):
        return self.__etalon_data
