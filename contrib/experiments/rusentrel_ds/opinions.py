from arekit.common.experiment.data_type import DataType
from arekit.common.labels.base import NeutralLabel
from arekit.common.opinions.collection import OpinionCollection
from arekit.contrib.experiments.rusentrel.opinions import RuSentrelOpinionOperations
from arekit.contrib.source.ruattitudes.news.helper import RuAttitudesNewsHelper
from arekit.contrib.source.rusentrel.io_utils import RuSentRelVersions


class RuSentrelWithRuAttitudesOpinionOperations(RuSentrelOpinionOperations):

    def __init__(self, data_io, rusetrel_version, experiments_name, neutral_annot_name, rusentrel_news_inds):
        assert(isinstance(rusentrel_news_inds, set))
        assert(isinstance(rusetrel_version, RuSentRelVersions))
        super(RuSentrelWithRuAttitudesOpinionOperations, self).__init__(
            data_io=data_io,
            neutral_annot_name=neutral_annot_name,
            rusentrel_news_ids=rusentrel_news_inds,
            experiment_name=experiments_name,
            version=rusetrel_version)

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

        if doc_id in self._rusentrel_news_ids:
            return super(RuSentrelWithRuAttitudesOpinionOperations, self).read_etalon_opinion_collection(doc_id)

        return OpinionCollection.init_as_custom(opinions=self.__get_opinions_in_news(doc_id),
                                                synonyms=self._data_io.SynonymsCollection)

    def read_neutral_opinion_collection(self, doc_id, data_type):
        assert(isinstance(doc_id, int))
        assert(isinstance(data_type, DataType))

        if doc_id not in self._rusentrel_news_ids and data_type == DataType.Train:
            return self.__get_opinions_in_news(doc_id=doc_id,
                                               opinion_check=lambda opinion: opinion.Sentiment == NeutralLabel())

        return super(RuSentrelWithRuAttitudesOpinionOperations, self).read_neutral_opinion_collection(
            doc_id=doc_id,
            data_type=data_type)
