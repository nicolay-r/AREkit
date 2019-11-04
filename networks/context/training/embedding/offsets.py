from core.common.embedding import Embedding


class TermsEmbeddingOffsets(object):
    """
    Describes indices distribution within a further TermsEmbedding.

    All parameters shifted by 1 because of a empty placeholder.
    """

    def __init__(self,
                 words_count,
                 missed_words_count,
                 tokens_count,
                 frames_count):
        assert(isinstance(words_count, int))
        assert(isinstance(missed_words_count, int))
        assert(isinstance(tokens_count, int))
        assert(isinstance(frames_count, int))
        self.__words_count = words_count
        self.__missed_words_count = missed_words_count
        self.__tokens_count = tokens_count
        self.__frames_count = frames_count

    # region properties

    @property
    def TotalCount(self):
        return 1 + \
               self.__missed_words_count + \
               self.__words_count + \
               self.__tokens_count + \
               self.__frames_count

    # endregion

    # region 'get' methods

    def get_word_index(self, index):
        return 1 + index

    def get_missed_word_index(self, index):
        return 1 + self.__words_count + index

    def get_token_index(self, index):
        return 1 + self.__words_count + self.__missed_words_count + index

    def get_frame_index(self, index):
        return 1 + self.__words_count + self.__missed_words_count + self.__tokens_count + index

    # endregion

    @staticmethod
    def iter_words_vocabulary(words_embedding, missed_words_embedding, tokens_embedding, frames_embedding):
        assert(isinstance(words_embedding, Embedding))
        assert(isinstance(missed_words_embedding, Embedding))
        assert(isinstance(tokens_embedding, Embedding))
        assert(isinstance(frames_embedding, Embedding))

        offsets = TermsEmbeddingOffsets(words_count=words_embedding.VocabularySize,
                                        missed_words_count=missed_words_embedding.VocabularySize,
                                        tokens_count=tokens_embedding.VocabularySize,
                                        frames_count=frames_embedding.VocabularySize)

        all_words = [(0, u'PADDING')]

        for word, index in words_embedding.iter_vocabulary():
            assert(isinstance(word, unicode))
            all_words.append((offsets.get_word_index(index), word))

        for missed_word, index in missed_words_embedding.iter_vocabulary():
            assert(isinstance(missed_word, unicode))
            all_words.append((offsets.get_missed_word_index(index), missed_word))

        for token, index in tokens_embedding.iter_vocabulary():
            assert(isinstance(token, unicode))
            all_words.append((offsets.get_token_index(index), token))

        for frame, index in frames_embedding.iter_vocabulary():
            assert(isinstance(frame, unicode))
            all_words.append((offsets.get_frame_index(index), frame))

        assert(len(all_words) == offsets.TotalCount)

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
