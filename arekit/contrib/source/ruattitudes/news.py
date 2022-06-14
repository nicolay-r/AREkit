from arekit.common.news.base import News
from arekit.contrib.source.ruattitudes.sentence import RuAttitudesSentence


class RuAttitudesNews(News):

    def __init__(self, sentences, news_index):
        assert(len(sentences) > 0)

        super(RuAttitudesNews, self).__init__(doc_id=news_index, sentences=sentences)

        self.__set_owners()
        self.__objects_before_sentence = self.__cache_objects_declared_before()

    # region properties

    @property
    def Title(self):
        return self._sentences[0]

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
