import numpy as np

from core.processing.text.tokens import Tokens
from core.common.utils import get_random_vector
from core.common.embedding import Embedding


# TODO: Move to ./embeddings/token.py
class TokenEmbedding(Embedding):
    """
    Embedding vectors for text punctuation, based on Tokens in parsed text
    """

    @classmethod
    def from_supported_tokens(cls, vector_size):
        matrix = []
        tokens_list = list(Tokens.iter_supported_tokens())

        for _ in tokens_list:
            matrix.append(get_random_vector(vector_size, seed=1))

        return cls(matrix=np.array(matrix),
                   words=tokens_list)
