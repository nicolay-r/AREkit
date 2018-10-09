from core.source.tokens import Tokens
from core.runtime.utils import get_random_vector


class TokenEmbeddingVectors:
    """
    Embedding vectors for text punctuation, based on Tokens
    """

    tokens = [Tokens.COMMA,
              Tokens.COLON,
              Tokens.SEMICOLON,
              Tokens.QUOTE,
              Tokens.DASH,
              Tokens.EXC_SIGN,
              Tokens.QUESTION_SIGN,
              Tokens.OPEN_BRACKET,
              Tokens.CLOSED_BRACKET,
              Tokens.LONG_DASH,
              Tokens.NUMBER,
              Tokens.TRIPLE_DOTS,
              Tokens.UNKNOWN_CHAR,
              Tokens.UNKNOWN_WORD]

    def __init__(self, vector_size):
        self.E = {}
        for index, token in enumerate(self.tokens):
            self.E[token] = get_random_vector(vector_size, index)

    @staticmethod
    def count():
        return len(TokenEmbeddingVectors.tokens)

    @staticmethod
    def get_token_index(token):
        assert(isinstance(token, unicode))
        return TokenEmbeddingVectors.tokens.index(token)

    def __getitem__(self, token):
        assert (isinstance(token, unicode))
        return self.E[token]

    def __iter__(self):
        for token in self.E:
            yield token

    def __contains__(self, token):
        assert (isinstance(token, unicode))
        return token in self.E
