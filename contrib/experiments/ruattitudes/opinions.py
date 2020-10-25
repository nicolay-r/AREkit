from arekit.common.experiment.data_type import DataType
from arekit.common.experiment.formats.opinions import OpinionOperations
from arekit.common.labels.base import NeutralLabel
from arekit.common.opinions.collection import OpinionCollection
from arekit.common.synonyms import SynonymsCollection
from arekit.contrib.source.ruattitudes.news.helper import RuAttitudesNewsHelper


class RuAttitudesOpinionOperations(OpinionOperations):

    def __init__(self, synonyms, neutral_root):
        assert(isinstance(synonyms, SynonymsCollection))

        super(RuAttitudesOpinionOperations, self).__init__()

        self._set_synonyms_collection(synonyms)
        self._set_neutral_root(neutral_root)

        self.__ru_attitudes = None

    def set_ru_attitudes(self, ra):
        assert(isinstance(ra, dict))
        self.__ru_attitudes = ra

    # region private methods

    def __get_opinions_in_news(self, doc_id, opinion_check=lambda _: True):
        news = self.__ru_attitudes[doc_id]
        return [opinion
                for opinion, _ in RuAttitudesNewsHelper.iter_opinions_with_related_sentences(news)
                if opinion_check(opinion)]

    # endregion

    def read_etalon_opinion_collection(self, doc_id):
        assert(isinstance(doc_id, int))

        return OpinionCollection.init_as_custom(opinions=self.__get_opinions_in_news(doc_id),
                                                synonyms=self._synonyms)

    def read_neutral_opinion_collection(self, doc_id, data_type):
        assert(isinstance(doc_id, int))
        assert(isinstance(data_type, DataType))

        if data_type == DataType.Train:
            return self.__get_opinions_in_news(doc_id=doc_id,
                                               opinion_check=lambda opinion: opinion.Sentiment == NeutralLabel())
