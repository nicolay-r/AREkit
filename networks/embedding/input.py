import logging
import numpy as np

from arekit.common.embeddings.base import Embedding
from arekit.common.embeddings.tokens import TokenEmbedding
from arekit.networks.embedding.offsets import TermsEmbeddingOffsets
from arekit.networks.debug import DebugKeys


logger = logging.getLogger(__name__)


def create_term_embedding_matrix(word_embedding,
                                 entity_embedding,
                                 token_embedding):
    """
    Compose complete embedding matrix, which includes:
        - word embeddings
        - entity embeddings
        - token embeddings

    returns: np.ndarray(words_count, embedding_size)
        embedding matrix which includes embedding both for words and
        entities
    """
    assert(isinstance(word_embedding, Embedding))
    assert(isinstance(entity_embedding, Embedding))
    assert(isinstance(token_embedding, TokenEmbedding))

    embedding_offsets = TermsEmbeddingOffsets(words_count=word_embedding.VocabularySize,
                                              entities_count=entity_embedding.VocabularySize,
                                              tokens_count=token_embedding.VocabularySize)
    matrix = np.zeros((embedding_offsets.TotalCount, word_embedding.VectorSize))

    # words.
    for word, index in word_embedding.iter_vocabulary():
        matrix[embedding_offsets.get_word_index(index)] = word_embedding.get_vector_by_index(index)

    # entities
    for word, index in entity_embedding.iter_vocabulary():
        matrix[embedding_offsets.get_entity_index(index)] = entity_embedding.get_vector_by_index(index)

    # tokens.
    for token_value, index in token_embedding.iter_vocabulary():
        matrix[embedding_offsets.get_token_index(index)] = token_embedding[token_value]

    if DebugKeys.DisplayTermEmbeddingParameters:
        logger.info("Term matrix shape: {}".format(matrix.shape))
        embedding_offsets.log_info()

    return matrix