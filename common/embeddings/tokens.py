import numpy as np

from core.processing.text.tokens import Tokens
from core.common.utils import get_random_uniform_with_fixed_seed
from core.common.embeddings.base import Embedding


class TokenEmbedding(Embedding):
    """
    Embedding vectors for text punctuation, based on Tokens in parsed text
    """

    @classmethod
    def from_supported_tokens(cls, vector_size):
        matrix = []
        tokens_list = list(Tokens.iter_supported_tokens())

        for token_index, _ in enumerate(tokens_list):
            matrix.append(get_random_uniform_with_fixed_seed(vector_size, seed=token_index))

        return cls(matrix=np.array(matrix),
                   words=tokens_list)
