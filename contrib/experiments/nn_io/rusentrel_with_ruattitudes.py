import logging

from arekit.networks.data_type import DataType
from arekit.common.opinions.collection import OpinionCollection
from arekit.contrib.experiments.nn_io.rusentrel import RuSentRelBasedNeuralNetworkIO
from arekit.contrib.experiments.nn_io.utils import read_ruattitudes_in_memory
from arekit.source.ruattitudes.helpers.news_helper import RuAttitudesNewsHelper
from arekit.source.ruattitudes.helpers.parsed_news import RuAttitudesParsedNewsHelper


logger = logging.getLogger(__name__)


class RuSentRelWithRuAttitudesBasedExperimentIO(RuSentRelBasedNeuralNetworkIO):
    """
    IO for the experiment with distant supervision for sentiment attitude extraction task.
    Paper: https://www.aclweb.org/anthology/R19-1118/
    """

    def __init__(self, model_name, data_io):
        super(RuSentRelWithRuAttitudesBasedExperimentIO, self).__init__(model_name=model_name,
                                                                        data_io=data_io)

        logger.debug("Loading RuAttitudes collection in memory, please wait ...")
        self.__ru_attitudes = read_ruattitudes_in_memory(data_io.Stemmer)

    # region 'read' public methods

    def read_parsed_news(self, doc_id):
        if doc_id in self.RuSentRelNewsIDsList:
            return super(RuSentRelWithRuAttitudesBasedExperimentIO, self).read_parsed_news(doc_id=doc_id)

        news = self.__ru_attitudes[doc_id]
        parsed_news = RuAttitudesParsedNewsHelper.create_parsed_news(doc_id=doc_id,
                                                                     news=news)

        return news, parsed_news

    def read_etalon_opinion_collection(self, doc_id):
        assert(isinstance(doc_id, int))

        if doc_id in self.RuSentRelNewsIDsList:
            return super(RuSentRelWithRuAttitudesBasedExperimentIO, self).read_etalon_opinion_collection(doc_id)

        news = self.__ru_attitudes[doc_id]
        opinions = [opinion for opinion, _ in RuAttitudesNewsHelper.iter_opinions_with_related_sentences(news)]

        return OpinionCollection(opinions=opinions,
                                 synonyms=self.DataIO.SynonymsCollection)

    def read_neutral_opinion_collection(self, doc_id, data_type):
        assert(isinstance(doc_id, int))
        assert(isinstance(data_type, unicode))

        if doc_id not in self.RuSentRelNewsIDsList:
            news = self.__ru_attitudes[doc_id]
            opinions, _ = RuAttitudesNewsHelper.iter_opinions_with_related_sentences(news=news)
            return opinions

        return super(RuSentRelWithRuAttitudesBasedExperimentIO, self).read_neutral_opinion_collection(
            doc_id=doc_id,
            data_type=data_type)

    def iter_news_indices(self, data_type):
        super(RuSentRelWithRuAttitudesBasedExperimentIO, self).iter_news_indices(data_type)

        if data_type == DataType.Train:
            for doc_id in self.__ru_attitudes.iterkeys():
                yield doc_id

    # endregion

