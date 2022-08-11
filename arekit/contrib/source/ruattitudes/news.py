from arekit.contrib.source.ruattitudes.sentence import RuAttitudesSentence


class RuAttitudesNews(object):

    def __init__(self, sentences, news_index):
        assert(len(sentences) > 0)

        self.__sentences = sentences
        self.__objects_before_sentence = self.__cache_objects_declared_before()
        self.__news_index = news_index

        self.__set_owners()

    # region properties

    @property
    def ID(self):
        return self.__news_index

    @property
    def Title(self):
        return self.__sentences[0]

    # endregion

    # region private methods

    def __set_owners(self):
        for sentence in self.__sentences:
            assert(isinstance(sentence, RuAttitudesSentence))
            sentence.set_owner(self)

    def __cache_objects_declared_before(self):
        d = {}
        before = 0
        for s in self.__sentences:
            assert(isinstance(s, RuAttitudesSentence))
            d[s.SentenceIndex] = before
            before += s.ObjectsCount

        return d

    # endregion

    def get_objects_declared_before(self, sentence_index):
        return self.__objects_before_sentence[sentence_index]

    def iter_sentences(self):
        for sentence in self.__sentences:
            yield sentence
