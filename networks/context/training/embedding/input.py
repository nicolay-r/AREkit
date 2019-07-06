import numpy as np

from core.networks.context.training.embedding.offsets import TermsEmbeddingOffsets
from core.source.embeddings.base import Embedding
from core.source.embeddings.tokens import TokenEmbedding
from core.networks.context.debug import DebugKeys


def create_term_embedding_matrix(word_embedding,
                                 missed_embedding,
                                 token_embedding,
                                 frame_embedding):
    """
    Compose complete embedding matrix, which includes:
        - word embeddings
        - token embeddings
        - frame embeddings

    word_embedding: Embedding
        embedding vocabulary for words
    missed_embedding: Embedding
        embedding for missed words of word_vocabulary
    returns: np.ndarray(words_count, embedding_size)
        embedding matrix which includes embedding both for words and
        entities
    """
    assert(isinstance(word_embedding, Embedding))
    assert(isinstance(missed_embedding, Embedding))
    assert(isinstance(token_embedding, TokenEmbedding))
    assert(isinstance(frame_embedding, Embedding))

    embedding_offsets = TermsEmbeddingOffsets(words_count=word_embedding.VocabularySize,
                                              missed_word_embedding=missed_embedding.VocabularySize,
                                              tokens_count=token_embedding.VocabularySize,
                                              frames_count=frame_embedding.VocabularySize)
    matrix = np.zeros((embedding_offsets.TotalCount, word_embedding.VectorSize))

    # words.
    for word, index in word_embedding.iter_vocabulary():
        matrix[embedding_offsets.get_word_index(index)] = word_embedding.get_vector_by_index(index)

    # missed words.
    for word, index in missed_embedding.iter_vocabulary():
        matrix[embedding_offsets.get_static_word_index(index)] = missed_embedding.get_vector_by_index(index)

    # tokens.
    for token_value, index in token_embedding.iter_vocabulary():
        matrix[embedding_offsets.get_token_index(index)] = token_embedding[token_value]

    # frames
    for frame_value, index in frame_embedding.iter_vocabulary():
        matrix[embedding_offsets.get_frame_index(index)] = frame_embedding[frame_value]

    if DebugKeys.DisplayTermEmbeddingParameters:
        print "Term matrix shape: {}".format(matrix.shape)
        embedding_offsets.debug_print()

    # used as a placeholder
    matrix[0] = 0

    return matrix