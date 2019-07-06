class TermsEmbeddingOffsets(object):
    """
    Describes indices distribution within a further TermsEmbedding.
    """

    def __init__(self,
                 words_count,
                 missed_word_embedding,
                 tokens_count,
                 frames_count):
        assert(isinstance(words_count, int))
        assert(isinstance(missed_word_embedding, int))
        assert(isinstance(tokens_count, int))
        assert(isinstance(frames_count, int))
        self.__words_count = words_count
        self.__missed_words_count = missed_word_embedding
        self.__tokens_count = tokens_count
        self.__frames_count = frames_count

    @property
    def TotalCount(self):
        return self.__missed_words_count + \
               self.__words_count + \
               self.__tokens_count + \
               self.__frames_count

    def get_word_index(self, index):
        return index

    def get_static_word_index(self, index):
        return self.__words_count + index

    def get_token_index(self, index):
        return self.__words_count + self.__missed_words_count + index

    def get_frame_index(self, index):
        return self.__words_count + self.__missed_words_count + self.__tokens_count + index

    def debug_print(self):
        print "Term embedding matrix details ..."
        print "\t\tWords count: {}".format(self.__words_count)
        print "\t\tMissed words count: {}".format(self.__missed_words_count)
        print "\t\tTokens count: {}".format(self.__tokens_count)
        print "\t\tFrames count: {}".format(self.__frames_count)
