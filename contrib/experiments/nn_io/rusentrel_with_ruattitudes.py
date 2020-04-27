import logging

from arekit.common.experiment.data_type import DataType
from arekit.common.opinions.collection import OpinionCollection

from arekit.contrib.experiments.rusentrel import RuSentRelBasedNeuralNetworkIO
from arekit.contrib.experiments.nn_io.utils import read_ruattitudes_in_memory

from arekit.source.ruattitudes.helpers.news_helper import RuAttitudesNewsHelper
from arekit.source.ruattitudes.helpers.parsed_news import RuAttitudesParsedNewsHelper


logger = logging.getLogger(__name__)


class RuSentRelWithRuAttitudesBasedExperimentIO(RuSentRelBasedNeuralNetworkIO):
    """
    IO for the experiment with distant supervision for sentiment attitude extraction task.
    Paper: https://www.aclweb.org/anthology/R19-1118/
    """

    def __init__(self, data_io, prepare_model_root):
        super(RuSentRelWithRuAttitudesBasedExperimentIO, self).__init__(data_io=data_io,
                                                                        prepare_model_root=prepare_model_root)

        logger.debug("Loading RuAttitudes collection in memory, please wait ...")
        self.__ru_attitudes = read_ruattitudes_in_memory(data_io.Stemmer)

    # region private methods

    def __get_opinions_in_news(self, doc_id):
        news = self.__ru_attitudes[doc_id]
        return [opinion for opinion, _ in RuAttitudesNewsHelper.iter_opinions_with_related_sentences(news)]

    # endregion

    # region 'read' public methods

    def read_parsed_news(self, doc_id):
        if doc_id in self.RuSentRelNewsIDsList:
            return super(RuSentRelWithRuAttitudesBasedExperimentIO, self).read_parsed_news(doc_id=doc_id)

        news = self.__ru_attitudes[doc_id]
        parsed_news = RuAttitudesParsedNewsHelper.create_parsed_news(doc_id=doc_id, news=news)

        return news, parsed_news

    def read_etalon_opinion_collection(self, doc_id):
        assert(isinstance(doc_id, int))

        if doc_id in self.RuSentRelNewsIDsList:
            return super(RuSentRelWithRuAttitudesBasedExperimentIO, self).read_etalon_opinion_collection(doc_id)

        return OpinionCollection(opinions=self.__get_opinions_in_news(doc_id),
                                 synonyms=self.DataIO.SynonymsCollection)

    def read_neutral_opinion_collection(self, doc_id, data_type):
        assert(isinstance(doc_id, int))
        assert(isinstance(data_type, unicode))

        if doc_id not in self.RuSentRelNewsIDsList:
            return self.__get_opinions_in_news(doc_id=doc_id)

        return super(RuSentRelWithRuAttitudesBasedExperimentIO, self).read_neutral_opinion_collection(
            doc_id=doc_id,
            data_type=data_type)

    def iter_news_indices(self, data_type):
        for doc_id in super(RuSentRelWithRuAttitudesBasedExperimentIO, self).iter_news_indices(data_type):
            yield doc_id

        if data_type == DataType.Train:
            for doc_id in self.__ru_attitudes.iterkeys():
                yield doc_id

    # endregion

