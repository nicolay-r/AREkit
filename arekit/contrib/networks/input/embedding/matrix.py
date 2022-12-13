import logging
import numpy as np

from arekit.contrib.networks.input.embedding.offsets import TermsEmbeddingOffsets
from arekit.contrib.networks.embedding import Embedding

logger = logging.getLogger(__name__)


def create_term_embedding_matrix(term_embedding):
    """
    Compose complete embedding matrix, which includes:
        - word embeddings
        - entity embeddings
        - token embeddings

    returns: np.ndarray(words_count, embedding_size)
        embedding matrix which includes embedding both for words and
        entities
    """
    assert(isinstance(term_embedding, Embedding))

    embedding_offsets = TermsEmbeddingOffsets(words_count=term_embedding.VocabularySize)
    matrix = np.zeros((embedding_offsets.TotalCount, term_embedding.VectorSize))

    for word, index in term_embedding.iter_vocabulary():
        matrix[embedding_offsets.get_word_index(index)] = term_embedding.get_vector_by_index(index)

    return matrix