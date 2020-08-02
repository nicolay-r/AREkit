from core.source.tokens import Tokens
from core.runtime.utils import get_random_vector


class TokenEmbeddingVectors:
    """
    Embedding vectors for text punctuation, based on Tokens
    """

    token_values = [Tokens.COMMA,
                    Tokens.COLON,
                    Tokens.SEMICOLON,
                    Tokens.QUOTE,
                    Tokens.DASH,
                    Tokens.DOT,
                    Tokens.EXC_SIGN,
                    Tokens.QUESTION_SIGN,
                    Tokens.OPEN_BRACKET,
                    Tokens.CLOSED_BRACKET,
                    Tokens.LONG_DASH,
                    Tokens.NUMBER,
                    Tokens.TRIPLE_DOTS,
                    Tokens.UNKNOWN_CHAR,
                    Tokens.NEW_LINE,
                    Tokens.URL,
                    Tokens.UNKNOWN_WORD]

    def __init__(self, vector_size):
        self.E = {}
        for index, token_value in enumerate(self.token_values):
            self.E[token_value] = get_random_vector(vector_size, index)

    @staticmethod
    def count():
        return len(TokenEmbeddingVectors.token_values)

    @staticmethod
    def get_token_index(token_value):
        assert(isinstance(token_value, str))
        return TokenEmbeddingVectors.token_values.index(token_value)

    def __getitem__(self, token_value):
        assert (isinstance(token_value, str))
        return self.E[token_value]

    def __iter__(self):
        for token_value in self.E:
            yield token_value

    def __contains__(self, token_value):
        assert (isinstance(token_value, str))
        return token_value in self.E
