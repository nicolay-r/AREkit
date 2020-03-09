import logging

import numpy as np

from arekit.common.embeddings.base import Embedding

logger = logging.getLogger(__name__)


def create_term_embedding(term,
                          embedding,
                          max_part_size=3,
                          word_separator=u' '):
    """
    Embedding algorithm based on parts (trigrams originally)
    """
    assert(isinstance(term, unicode))
    assert(isinstance(embedding, Embedding))

    count = 0
    vector = np.zeros(embedding.VectorSize)
    for word in term.split(word_separator):
        v, c = __create_embedding_for_word(word=word,
                                           max_part_size=max_part_size,
                                           embedding=embedding)
        count += c
        vector = vector + v

    return vector / count if count > 0 else vector


def __create_embedding_for_word(word, max_part_size, embedding):
    assert(isinstance(word, unicode))
    assert(isinstance(embedding, Embedding))

    if word in embedding:
        return embedding[word], 1

    c_i = 0
    c_l = max_part_size
    count = 0
    vector = np.zeros(embedding.VectorSize)
    missings = []

    while c_i < len(word):

        if c_l == 0:
            missings.append(c_i)
            c_i += 1
            c_l = max_part_size
            continue

        right_b = min(len(word), c_i + c_l)

        s_i = embedding.contains_as_plain(word=word[c_i:right_b])

        if s_i is None:
            c_l -= 1
            continue

        vector = vector + embedding.get_vector_by_index(s_i)
        c_i += c_l
        count += 1

    w_debug = u''.join([u'?' if i in missings else ch
                        for i, ch in enumerate(word)])
    logger.debug(u'Embedded: {}'.format(w_debug).encode('utf-8'))

    return vector, count
