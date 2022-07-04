class BasePairToCompare(object):
    """ Document-related pair to compare.
    """

    def __init__(self, doc_id):
        assert(isinstance(doc_id, int))
        self.__doc_id = doc_id

    @property
    def TestData(self):
        raise NotImplementedError()

    @property
    def EtalonData(self):
        raise NotImplementedError()

    @property
    def DocumentID(self):
        return self.__doc_id
