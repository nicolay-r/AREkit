import logging
import numpy as np

from arekit.networks.context.embedding.offsets import TermsEmbeddingOffsets
from arekit.common.embeddings.base import Embedding
from arekit.common.embeddings.tokens import TokenEmbedding
from arekit.networks.context.debug import DebugKeys


logger = logging.getLogger(__name__)


def create_term_embedding_matrix(word_embedding,
                                 custom_embedding,
                                 token_embedding,
                                 frame_embedding):
    """
    Compose complete embedding matrix, which includes:
        - word embeddings
        - token embeddings
        - frame embeddings

    word_embedding: Embedding
        embedding vocabulary for words
    custom_embedding: Embedding
        embedding for custom words of word_vocabulary
    returns: np.ndarray(words_count, embedding_size)
        embedding matrix which includes embedding both for words and
        entities
    """
    assert(isinstance(word_embedding, Embedding))
    assert(isinstance(custom_embedding, Embedding))
    assert(isinstance(token_embedding, TokenEmbedding))
    assert(isinstance(frame_embedding, Embedding))

    embedding_offsets = TermsEmbeddingOffsets(words_count=word_embedding.VocabularySize,
                                              custom_words_count=custom_embedding.VocabularySize,
                                              tokens_count=token_embedding.VocabularySize,
                                              frames_count=frame_embedding.VocabularySize)
    matrix = np.zeros((embedding_offsets.TotalCount, word_embedding.VectorSize))

    # words.
    for word, index in word_embedding.iter_vocabulary():
        matrix[embedding_offsets.get_word_index(index)] = word_embedding.get_vector_by_index(index)

    # custom words.
    for word, index in custom_embedding.iter_vocabulary():
        matrix[embedding_offsets.get_custom_word_index(index)] = custom_embedding.get_vector_by_index(index)

    # tokens.
    for token_value, index in token_embedding.iter_vocabulary():
        matrix[embedding_offsets.get_token_index(index)] = token_embedding[token_value]

    # frames
    for frame_value, index in frame_embedding.iter_vocabulary():
        matrix[embedding_offsets.get_frame_index(index)] = frame_embedding[frame_value]

    if DebugKeys.DisplayTermEmbeddingParameters:
        logger.info("Term matrix shape: {}".format(matrix.shape))
        embedding_offsets.log_info()

    return matrix