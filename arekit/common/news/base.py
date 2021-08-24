class News(object):

    def __init__(self, news_id, sentences):
        assert(isinstance(news_id, int))
        assert(isinstance(sentences, list))

        self.__news_id = news_id
        self._sentences = sentences

    # region properties

    @property
    def ID(self):
        return self.__news_id

    @property
    def SentencesCount(self):
        """ Provides total amount of sentences within a news
            At present is useful for:
            -   CV-splitters, which may rely on sentences count.
            -   Text parsing.
        """
        return len(self._sentences)

    # endregion

    def sentence_to_terms_list(self, sent_ind):
        assert(isinstance(sent_ind, int))
        sentence = self._sentences[sent_ind]
        return self._sentence_to_terms_list_core(sentence)

    def iter_sentences(self, return_text):
        """
        return: list of string
            iterates per raw sentences.
        """
        assert(isinstance(return_text, bool))

        for sentence in self._sentences:
            if return_text:
                yield sentence.Text
            else:
                yield sentence

    def extract_linked_text_opinions(self, opinion):
        """
        opinions: iterable Opinion
            is an iterable opinions that should be used to find a related text_opinion entries.
        """
        raise NotImplementedError()

    def get_entities_collection(self):
        raise NotImplementedError("Document does not support entities collection generation.")

    @staticmethod
    def _sentence_to_terms_list_core(sentence):
        """
        pipeline processing application towards the particular sentence.
        """
        raise NotImplementedError()
