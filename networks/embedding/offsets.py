import logging
from arekit.common.embeddings.base import Embedding


logger = logging.getLogger(__name__)


class TermsEmbeddingOffsets(object):
    """
    Describes indices distribution within a further TermsEmbedding.

    All parameters shifted by 1 because of a empty placeholder.
    """

    def __init__(self,
                 words_count,
                 entities_count,
                 tokens_count):
        assert(isinstance(words_count, int))
        assert(isinstance(entities_count, int))
        assert(isinstance(tokens_count, int))
        self.__words_count = words_count
        self.__entities_count = entities_count
        self.__tokens_count = tokens_count

    # region properties

    @property
    def TotalCount(self):
        return 1 + \
               self.__entities_count + \
               self.__words_count + \
               self.__tokens_count

    # endregion

    # region 'get' methods

    def get_word_index(self, index):
        return 1 + index

    def get_entity_index(self, index):
        return 1 + self.__words_count + index

    def get_token_index(self, index):
        return 1 + self.__words_count + self.__entities_count + index

    # endregion

    @staticmethod
    def iter_words_vocabulary(words_embedding, entities_embedding, tokens_embedding):
        assert(isinstance(words_embedding, Embedding))
        assert(isinstance(entities_embedding, Embedding))
        assert(isinstance(tokens_embedding, Embedding))

        offsets = TermsEmbeddingOffsets(words_count=words_embedding.VocabularySize,
                                        entities_count=entities_embedding.VocabularySize,
                                        tokens_count=tokens_embedding.VocabularySize)

        all_words = [(0, u'PADDING')]

        for word, index in words_embedding.iter_vocabulary():
            assert(isinstance(word, unicode))
            all_words.append((offsets.get_word_index(index), word))

        for custom_word, index in entities_embedding.iter_vocabulary():
            assert(isinstance(custom_word, unicode))
            all_words.append((offsets.get_entity_index(index), custom_word))

        for token, index in tokens_embedding.iter_vocabulary():
            assert(isinstance(token, unicode))
            all_words.append((offsets.get_token_index(index), token))

        assert(len(all_words) == offsets.TotalCount)

        for key, word in sorted(all_words, key=lambda item: item[0]):
            yield word

    # region debug methods

    def log_info(self):
        logger.info("Term embedding matrix details ...")
        logger.info("\t\tWords count: {}".format(self.__words_count))
        logger.info("\t\tEntities count: {}".format(self.__entities_count))
        logger.info("\t\tTokens count: {}".format(self.__tokens_count))

    # endregion
