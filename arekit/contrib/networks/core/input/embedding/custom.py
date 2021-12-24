import logging
import numpy as np
from arekit.contrib.networks.embeddings.base import Embedding

logger = logging.getLogger(__name__)


def create_term_embedding(term, embedding, check, word_separator=' '):
    """
    Embedding algorithm based on parts (trigrams originally)
    """
    assert(isinstance(term, str))
    assert(isinstance(embedding, Embedding))

    if term in embedding:
        word, embedding = __get_from_embedding(term=term, embedding=embedding)
    else:
        word, embedding = __compose_from_parts(term=term,
                                               embedding=embedding,
                                               check=check,
                                               word_separator=word_separator)

    # In order to prevent a problem of the further separations during reading process.
    # it is necessary to replace the separators with the other chars.
    word = word.replace(word_separator, '-')

    return word, embedding


def __compose_from_parts(term, embedding, check, word_separator, max_part_size=3, do_lowercase=True):
    # remove empty spaces before and after.
    term = term.strip()

    if check:
        if ' ' in term:
            print("FAILED: [{}]".format(term))

    # perform lowercasing
    if do_lowercase:
        term = term.lower()

    # Calculating vector from term parts
    count = 0
    vector = np.zeros(embedding.VectorSize)
    for word in term.split(word_separator):
        v, c = __create_embedding_for_word(word=word,
                                           max_part_size=max_part_size,
                                           embedding=embedding)
        count += c
        vector = vector + v

    return term, vector / count if count > 0 else vector


def __get_from_embedding(term, embedding):
    if ' ' in term:
        print("EXISTED IN EMBEDDING: [{}]".format(term))
    return embedding.try_get_related_word(term), embedding[term]


def __create_embedding_for_word(word, max_part_size, embedding):
    assert(isinstance(word, str))
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

        s_i = embedding.try_find_index_by_plain_word(word=word[c_i:right_b])

        if s_i is None:
            c_l -= 1
            continue

        vector += embedding.get_vector_by_index(s_i)
        c_i += c_l
        count += 1

    debug = False
    if debug:
        w_debug = ''.join(['?' if i in missings else ch
                            for i, ch in enumerate(word)])
        logger.debug('Embedded: {}'.format(w_debug).encode('utf-8'))

    return vector, count
