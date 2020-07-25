import logging
from arekit.common.embeddings.base import Embedding


logger = logging.getLogger(__name__)


class TermsEmbeddingOffsets(object):
    """
    Describes indices distribution within a further TermsEmbedding.

    All parameters shifted by 1 because of a empty placeholder.
    """

    def __init__(self, words_count):
        assert(isinstance(words_count, int))
        self.__words_count = words_count

    # region properties

    @property
    def TotalCount(self):
        return 1 + self.__words_count

    # endregion

    # region 'get' methods

    def get_word_index(self, index):
        return 1 + index

    # endregion

    @staticmethod
    def extract_vocab(words_embedding):
        assert(isinstance(words_embedding, Embedding))

        offsets = TermsEmbeddingOffsets(words_count=words_embedding.VocabularySize)

        all_words = [(0, u'PADDING')]

        for word, index in words_embedding.iter_vocabulary():
            assert(isinstance(word, unicode))
            all_words.append((offsets.get_word_index(index), word))

        assert(len(all_words) == offsets.TotalCount)

        for key, word in sorted(all_words, key=lambda item: item[0]):
            yield key, word

    # region debug methods

    def log_info(self):
        logger.info("Term embedding matrix details ...")
        logger.info("\t\tWords count: {}".format(self.__words_count))

    # endregion
