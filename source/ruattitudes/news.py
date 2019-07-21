from core.source.ruattitudes.sentence import Sentence


class News(object):

    def __init__(self, sentences, news_index):
        assert(isinstance(sentences, list))
        assert(len(sentences) > 0)
        assert(isinstance(news_index, int))
        self.__sentences = sentences
        self.__news_index = news_index
        self.__set_owners()

    @property
    def Title(self):
        return self.__sentences[0]

    @property
    def NewsIndex(self):
        return self.__news_index

    def __set_owners(self):
        for sentence in self.__sentences:
            assert(isinstance(sentence, Sentence))
            sentence.set_owner(self)

    def get_sentence(self, index):
        return self.__sentences[index]

    def iter_sentences(self):
        for sentence in self.__sentences:
            yield sentence
