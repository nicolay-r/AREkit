import numpy as np

from arekit.contrib.networks.embedding import Embedding
from arekit.contrib.utils.processing.text.tokens import Tokens


class TokenEmbedding(Embedding):
    """ Embedding vectors for text punctuation, based on Tokens in parsed text
    """

    @classmethod
    def from_supported_tokens(cls, vector_size, random_vector_func):
        """
        random_vector_func: func
            function with parameters (vector_size, seed)
        """
        assert(isinstance(vector_size, int))
        assert(callable(random_vector_func))

        matrix = []
        tokens_list = list(Tokens.iter_supported_tokens())

        for token_index, _ in enumerate(tokens_list):

            vector = random_vector_func(vector_size, token_index)

            matrix.append(vector)

        return cls(matrix=np.array(matrix),
                   words=tokens_list)
