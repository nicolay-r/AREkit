import numpy as np
from core.source.embeddings.base import Embedding


# TODO. Use default class.
class StaticEmbedding(Embedding):

    @classmethod
    def from_list_with_embedding_func(cls, words, embedding_func):
        assert(isinstance(words, list))
        assert(callable(embedding_func))

        matrix = []
        for word in words:
            matrix.append(embedding_func(word))

        return cls(matrix=np.array(matrix),
                   words=words)
