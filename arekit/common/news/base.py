class News(object):

    def __init__(self, doc_id, sentences):
        assert(isinstance(doc_id, int))
        assert(isinstance(sentences, list))

        self.__id = doc_id
        self._sentences = sentences

    # region properties

    @property
    def ID(self):
        return self.__id

    @property
    def SentencesCount(self):
        """ Provides total amount of sentences within a news
            At present is useful for:
            -   CV-splitters, which may rely on sentences count.
            -   Text parsing.
        """
        return len(self._sentences)

    # endregion

    def iter_sentences(self):
        for sentence in self._sentences:
            yield sentence

    def get_sentence(self, s_ind):
        return self._sentences[s_ind]
