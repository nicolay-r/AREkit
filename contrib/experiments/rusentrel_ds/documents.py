from arekit.common.experiment.data_type import DataType
from arekit.contrib.experiments.rusentrel.documents import RuSentrelDocumentOperations


class RuSentrelWithRuAttitudesDocumentOperations(RuSentrelDocumentOperations):

    def __init__(self, data_io, rusentrel_news_inds):
        assert(isinstance(rusentrel_news_inds, set))
        super(RuSentrelWithRuAttitudesDocumentOperations, self).__init__(data_io=data_io)
        self.__rusentrel_news = rusentrel_news_inds
        self.__ru_attitudes = None

    def set_ru_attitudes(self, ra):
        assert(isinstance(ra, dict))
        self.__ru_attitudes = ra

    def read_news(self, doc_id):
        if doc_id in self.__rusentrel_news:
            return super(RuSentrelWithRuAttitudesDocumentOperations, self).read_news(doc_id=doc_id)
        return self.__ru_attitudes[doc_id]

    def iter_news_indices(self, data_type):
        for doc_id in super(RuSentrelWithRuAttitudesDocumentOperations, self).iter_news_indices(data_type):
            yield doc_id

        if data_type == DataType.Train:
            for doc_id in self.__ru_attitudes.iterkeys():
                yield doc_id


