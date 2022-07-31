from urllib.parse import urlparse
from arekit.common.context.token import Token


# TODO. Provide the base (BaseTokens) type.
# TODO. With the related API at BaseTokens.
class Tokens:
    """
    Tokens used to describe a non-word text units, such as punctuation,
    uknown words/chars, smiles, etc.
    """

    _wrapper = "<[{}]>"
    COMMA = _wrapper.format(',')
    SEMICOLON = _wrapper.format(';')
    COLON = _wrapper.format(':')
    QUOTE = _wrapper.format('QUOTE')
    DASH = _wrapper.format('-')
    LONG_DASH = _wrapper.format('long_dash')
    DOT = _wrapper.format('.')
    TRIPLE_DOTS = _wrapper.format('…')
    EXC_SIGN = _wrapper.format('!')
    QUESTION_SIGN = _wrapper.format('?')
    OPEN_BRACKET = _wrapper.format('OPEN_BRACKET')
    CLOSED_BRACKET = _wrapper.format('CLOSED_BRACKET')
    NUMBER = _wrapper.format('NUMBER')
    NEW_LINE = _wrapper.format("NEW_LINE")
    UNKNOWN_CHAR = _wrapper.format('UNKNOWN_CHAR')
    UNKNOWN_WORD = _wrapper.format('UNKNOWN_WORD')
    URL = _wrapper.format("URL")

    __token_mapping = {
        ',': COMMA,
        '.': DOT,
        '…': TRIPLE_DOTS,
        ':': COLON,
        ';': SEMICOLON,
        '-': DASH,
        '—': LONG_DASH,
        '?': QUESTION_SIGN,
        '!': EXC_SIGN,
        '(': OPEN_BRACKET,
        ')': CLOSED_BRACKET,
        '{': OPEN_BRACKET,
        '}': CLOSED_BRACKET,
        '[': OPEN_BRACKET,
        ']': CLOSED_BRACKET,
        '\n': NEW_LINE,
        '«': QUOTE,
        '»': QUOTE,
        '"': QUOTE,
    }

    __supported_tokens = {
        COMMA,
        SEMICOLON,
        COLON,
        QUOTE,
        DASH,
        DOT,
        LONG_DASH,
        TRIPLE_DOTS,
        EXC_SIGN,
        QUESTION_SIGN,
        OPEN_BRACKET,
        CLOSED_BRACKET,
        NUMBER,
        URL,
        NEW_LINE,
        UNKNOWN_CHAR,
        UNKNOWN_WORD}

    @staticmethod
    def try_create(subterm):
        """
        Trying create a token by given 'term' parameter
        subterm: unicode
           I.e. term ending, so means a part of original term
        """
        assert(isinstance(subterm, str))
        if subterm not in Tokens.__token_mapping:
            return None
        return Token(term=subterm, token_value=Tokens.__token_mapping[subterm])

    @staticmethod
    def try_parse(term):
        assert(isinstance(term, str))
        for origin, token_value in Tokens.__token_mapping.items():
            if term == token_value:
                return Token(term=origin, token_value=token_value)

    @staticmethod
    def try_create_number(term):
        assert(isinstance(term, str))
        if not term.isdigit():
            return None
        return Token(term=term, token_value=Tokens.NUMBER)

    @staticmethod
    def try_create_url(term):
        assert(isinstance(term, str))
        result = urlparse(term)
        is_correct = result.scheme and result.netloc and result.path
        if not is_correct:
            return None
        return Token(term=term, token_value=Tokens.URL)

    @staticmethod
    def is_token(term):
        assert(isinstance(term, str))
        return term in Tokens.__supported_tokens

    @staticmethod
    def iter_chars_by_token(term):
        """
        Iterate through charts that is related to term
        token: char
        """
        assert(isinstance(term, str))
        for char, token in Tokens.__token_mapping.items():
            if term == token:
                yield char

    @staticmethod
    def iter_supported_tokens():
        for token in Tokens.__supported_tokens:
            yield token
