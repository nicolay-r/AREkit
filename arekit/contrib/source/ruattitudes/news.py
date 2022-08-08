from arekit.contrib.source.brat.news import BratNews
from arekit.contrib.source.ruattitudes.opinions.base import SentenceOpinion
from arekit.contrib.source.ruattitudes.sentence import RuAttitudesSentence


class RuAttitudesNews(BratNews):

    def __init__(self, sentences, news_index):
        assert(len(sentences) > 0)

        super(RuAttitudesNews, self).__init__(doc_id=news_index,
                                              sentences=sentences,
                                              text_opinions=None)

        self.__set_owners()
        self.__sentences = sentences
        self.__objects_before_sentence = self.__cache_objects_declared_before()

    # region properties

    @property
    def Title(self):
        return self._sentences[0]

    @property
    def TextOpinions(self):
        for sentence in self.__sentences:
            assert(isinstance(sentence, RuAttitudesSentence))
            for sentence_opinion in sentence.iter_sentence_opins():
                assert(isinstance(sentence_opinion, SentenceOpinion))
                yield sentence_opinion

    # endregion

    # region private methods

    def __set_owners(self):
        for sentence in self._sentences:
            assert(isinstance(sentence, RuAttitudesSentence))
            sentence.set_owner(self)

    def __cache_objects_declared_before(self):
        d = {}
        before = 0
        for s in self._sentences:
            assert(isinstance(s, RuAttitudesSentence))
            d[s.SentenceIndex] = before
            before += s.ObjectsCount

        return d

    # endregion

    def get_objects_declared_before(self, sentence_index):
        return self.__objects_before_sentence[sentence_index]
