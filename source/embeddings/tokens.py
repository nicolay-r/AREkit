from core.processing.text.tokens import Tokens
from core.common.utils import get_random_vector


class TokenEmbeddingVectors:
    """
    Embedding vectors for text punctuation, based on Tokens in parsed text
    """

    # TODO. Use default ctor.
    def __init__(self, vector_size):
        self.E = {}
        for index, token_value in enumerate(Tokens.iter_supported_tokens()):
            self.E[token_value] = get_random_vector(vector_size, index)

    # TODO. Provide classmethod.

    # TODO. Remove. Use non-static VocabularySize instead.
    @staticmethod
    def count():
        return len(TokenEmbeddingVectors.token_values)

    @staticmethod
    def get_token_index(token_value):
        assert(isinstance(token_value, unicode))
        return TokenEmbeddingVectors.token_values.index(token_value)

    def __getitem__(self, token_value):
        assert (isinstance(token_value, unicode))
        return self.E[token_value]

    def __contains__(self, token_value):
        assert (isinstance(token_value, unicode))
        return token_value in self.E
