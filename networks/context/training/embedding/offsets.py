from core.common.embedding import Embedding


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

    @staticmethod
    def iter_words_vocabulary(words_embedding, missed_words_embedding, tokens_embedding, frames_embedding):
        assert(isinstance(words_embedding, Embedding))
        assert(isinstance(missed_words_embedding, Embedding))
        assert(isinstance(tokens_embedding, Embedding))
        assert(isinstance(frames_embedding, Embedding))

        offsets = TermsEmbeddingOffsets(words_count=words_embedding.VocabularySize,
                                        missed_word_embedding=missed_words_embedding.VocabularySize,
                                        tokens_count=tokens_embedding.VocabularySize,
                                        frames_count=frames_embedding.VocabularySize)

        all_words = []

        for index, w in words_embedding.iter_vocabulary():
            all_words.append((offsets.get_word_index(index), w))

        for index, m_w in missed_words_embedding.iter_vocabulary():
            all_words.append((offsets.get_word_index(index), m_w))

        for index, token in tokens_embedding.iter_vocabulary():
            all_words.append((offsets.get_word_index(index), token))

        for index, frame in frames_embedding.iter_vocabulary():
            all_words.append((offsets.get_word_index(index), frame))

        for key, word in sorted(all_words, key=lambda item: item[0]):
            yield word

    # region debug methods

    def debug_print(self):
        print "Term embedding matrix details ..."
        print "\t\tWords count: {}".format(self.__words_count)
        print "\t\tMissed words count: {}".format(self.__missed_words_count)
        print "\t\tTokens count: {}".format(self.__tokens_count)
        print "\t\tFrames count: {}".format(self.__frames_count)

    # endregion
