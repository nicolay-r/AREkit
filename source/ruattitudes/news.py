class ProcessedNews(object):

    def __init__(self, processed_sentences, news_index):
        assert(isinstance(processed_sentences, list))
        assert(len(processed_sentences) > 0)
        assert(isinstance(news_index, int))
        self.__processed_sentences = processed_sentences
        self.__news_index = news_index

    @property
    def Title(self):
        return self.__processed_sentences[0]

    @property
    def NewsIndex(self):
        return self.__news_index

    def get_sentence(self, index):
        return self.__processed_sentences[index]

    def iter_processed_sentences(self):
        for s in self.__processed_sentences:
            yield s
