from arekit.common.evaluation.pairs.base import BasePairToCompare


class SingleDocumentDataPairsToCompare(BasePairToCompare):
    """ A single document pair provider.
    """

    def __init__(self, etalon_data, test_data):
        super(SingleDocumentDataPairsToCompare, self).__init__(doc_id=0)

        self.__test_data = test_data
        self.__etalon_data = etalon_data

    @property
    def TestData(self):
        return self.__test_data

    @property
    def EtalonData(self):
        return self.__etalon_data
