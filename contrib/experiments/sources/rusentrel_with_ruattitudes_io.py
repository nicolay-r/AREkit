from arekit.common.opinions.collection import OpinionCollection
from arekit.contrib.experiments.sources.utils import read_ruattitudes_in_memory
from arekit.processing.lemmatization.base import Stemmer
from arekit.source.ruattitudes.helpers.news_helper import RuAttitudesNewsHelper
from arekit.source.ruattitudes.helpers.parsed_news import RuAttitudesParsedNewsHelper
from arekit.contrib.experiments.sources.rusentrel_io import RuSentRelNetworkIO


class RuSentRelWithRuAttitudesIO(RuSentRelNetworkIO):
    """
    IO for the experiment with distant supervision for sentiment attitude extraction task.
    Paper: https://www.aclweb.org/anthology/R19-1118/
    """

    def __init__(self, model_name, experiments_io, cv_count):
        super(RuSentRelWithRuAttitudesIO, self).__init__(model_name=model_name,
                                                         experiments_io=experiments_io,
                                                         cv_count=cv_count)
        self.__ru_attitudes = None

    # region 'read' public methods

    def init_synonyms_collection(self, stemmer):
        assert(isinstance(stemmer, Stemmer))
        super(RuSentRelWithRuAttitudesIO, self).init_synonyms_collection(stemmer)

        # TODO. Replace with logger.
        print "Loading RuAttitudes collection in memory, please wait ..."

        self.__ru_attitudes = read_ruattitudes_in_memory(stemmer)

    def read_parsed_news(self, doc_id, keep_tokens, stemmer):
        if doc_id in self.RuSentRelNewsIDsList:
            return super(RuSentRelWithRuAttitudesIO, self).read_parsed_news(doc_id=doc_id,
                                                                            keep_tokens=keep_tokens,
                                                                            stemmer=stemmer)

        news = self.__ru_attitudes[doc_id]
        parsed_news = RuAttitudesParsedNewsHelper.create_parsed_news(doc_id=doc_id,
                                                                     news=news)

        return news, parsed_news

    def read_etalon_opinion_collection(self, doc_id):
        assert(isinstance(doc_id, int))

        if doc_id in self.RuSentRelNewsIDsList:
            return super(RuSentRelWithRuAttitudesIO, self).read_etalon_opinion_collection(doc_id)

        news = self.__ru_attitudes[doc_id]
        opinions = [opinion for opinion, _ in RuAttitudesNewsHelper.iter_opinions_with_related_sentences(news)]

        return OpinionCollection(opinions=opinions,
                                 synonyms=self.SynonymsCollection)

    def read_neutral_opinion_collection(self, doc_id, data_type):
        assert(isinstance(doc_id, int))
        assert(isinstance(data_type, unicode))

        if doc_id in self.RuSentRelNewsIDsList:
            return super(RuSentRelWithRuAttitudesIO, self).read_neutral_opinion_collection(
                doc_id=doc_id,
                data_type=data_type)

    def iter_train_data_indices(self):
        for doc_id in super(RuSentRelWithRuAttitudesIO, self).iter_train_data_indices():
            yield doc_id
        for doc_id in self.__ru_attitudes.iterkeys():
            yield doc_id

    # endregion

