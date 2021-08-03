class OpinionCollectionsToCompare(object):

    def __init__(self,
                 doc_id,
                 read_etalon_collection_func,
                 read_result_collection_func):
        assert(isinstance(doc_id, int))
        assert(callable(read_etalon_collection_func))
        assert(callable(read_result_collection_func))

        self.__test_opinions = read_result_collection_func(doc_id)
        self.__etalon_opinions = read_etalon_collection_func(doc_id)
        self.__doc_id = doc_id

    @property
    def TestOpinionCollection(self):
        return self.__test_opinions

    @property
    def EtalonOpinionCollection(self):
        return self.__etalon_opinions

    @property
    def DocumentID(self):
        return self.__doc_id