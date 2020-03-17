from arekit.common.opinions.collection import OpinionCollection
from arekit.processing.lemmatization.base import Stemmer
from arekit.source.ruattitudes.helpers.news_helper import RuAttitudesNewsHelper
from arekit.source.ruattitudes.helpers.parsed_news import RuAttitudesParsedNewsHelper
from arekit.source.ruattitudes.news import RuAttitudesNews
from arekit.source.ruattitudes.reader import RuAttitudesFormatReader
from readers.rusentrel_io import RuSentRelDataIO


class RuSentRelWithRuAttitudesDataIO(RuSentRelDataIO):
    """
    RuAttitudes reader additionally for RuSentRel data reader.
    """

    def __init__(self, stemmer, synonyms, model_name=u'bert-ds'):
        super(RuSentRelWithRuAttitudesDataIO, self).__init__(
            model_name=model_name,
            synonyms=synonyms)

        print "Loading RuAttitudes collection in memory, please wait ..."
        self.__ru_attitudes = self.__read_ruattitudes_in_memory(stemmer)

    # region 'read' public methods

    def read_document(self, doc_id, keep_tokens):
        if doc_id in self.RuSentRelNewsIDsList:
            return super(RuSentRelWithRuAttitudesDataIO, self).read_document(doc_id=doc_id,
                                                                             keep_tokens=keep_tokens)

        news = self.__ru_attitudes[doc_id]
        parsed_news = RuAttitudesParsedNewsHelper.create_parsed_news(doc_id=doc_id,
                                                                     news=news)

        return news, parsed_news

    def read_etalon_opinion_collection(self, doc_id):
        assert (isinstance(doc_id, int))

        if doc_id in self.RuSentRelNewsIDsList:
            return super(RuSentRelWithRuAttitudesDataIO, self).read_etalon_opinion_collection(doc_id)

        news = self.__ru_attitudes[doc_id]
        opinions = [opinion for opinion, _ in RuAttitudesNewsHelper.iter_opinions_with_related_sentences(news)]

        return OpinionCollection(opinions=opinions,
                                 synonyms=self.SynonymsCollection)

    def read_neutral_opinion_collection(self, doc_id, data_type):
        assert (isinstance(doc_id, int))
        assert (isinstance(data_type, unicode))

        if doc_id in self.RuSentRelNewsIDsList:
            return super(RuSentRelWithRuAttitudesDataIO, self).read_neutral_opinion_collection(doc_id, data_type)

        # TODO. Complete.
        pass

    # endregion

    # region private methods

    @staticmethod
    def __read_ruattitudes_in_memory(stemmer):
        assert (isinstance(stemmer, Stemmer))

        d = {}
        for news in RuAttitudesFormatReader.iter_news(stemmer=stemmer):
            assert (isinstance(news, RuAttitudesNews))
            d[news.NewsIndex] = news

        return d

    def iter_train_data_indices(self):
        for doc_id in super(RuSentRelWithRuAttitudesDataIO, self).iter_train_data_indices():
            yield doc_id
        for doc_id in self.__ru_attitudes.iterkeys():
            yield doc_id

    # endregion
