from arekit.common.experiment.data_type import DataType
from arekit.common.opinions.collection import OpinionCollection
from arekit.contrib.experiments.rusentrel.opinions import RuSentrelOpinionOperations
from arekit.source.ruattitudes.news.helper import RuAttitudesNewsHelper


class RuSentrelWithRuAttitudesOpinionOperations(RuSentrelOpinionOperations):

    def __init__(self, data_io, annot_name_func, rusentrel_news_inds):
        assert(isinstance(rusentrel_news_inds, set))
        super(RuSentrelWithRuAttitudesOpinionOperations, self).__init__(
            data_io=data_io,
            annot_name_func=annot_name_func,
            rusentrel_news_ids=rusentrel_news_inds)

        self.__ru_attitudes = None

    def set_ru_attitudes(self, ra):
        assert(isinstance(ra, dict))
        self.__ru_attitudes = ra

    # region private methods

    def __get_opinions_in_news(self, doc_id):
        news = self.__ru_attitudes[doc_id]
        return [opinion for opinion, _ in RuAttitudesNewsHelper.iter_opinions_with_related_sentences(news)]

    # endregion

    def read_etalon_opinion_collection(self, doc_id):
        assert(isinstance(doc_id, int))

        if doc_id in self._rusentrel_news_ids:
            return super(RuSentrelWithRuAttitudesOpinionOperations, self).read_etalon_opinion_collection(doc_id)

        return OpinionCollection(opinions=self.__get_opinions_in_news(doc_id),
                                 synonyms=self._data_io.SynonymsCollection)

    def read_neutral_opinion_collection(self, doc_id, data_type):
        assert(isinstance(doc_id, int))
        assert(isinstance(data_type, DataType))

        if doc_id not in self._rusentrel_news_ids:
            return self.__get_opinions_in_news(doc_id=doc_id)

        return super(RuSentrelWithRuAttitudesOpinionOperations, self).read_neutral_opinion_collection(
            doc_id=doc_id,
            data_type=data_type)

