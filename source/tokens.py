# -*- coding: utf-8 -*-


class Tokens:
    """
    Tokens used to describe a non-word text units, such as punctuation,
    uknown words/chars, smiles, etc.
    """

    _wrapper = u"<[{}]>"
    COMMA = _wrapper.format(u',')
    SEMICOLON = _wrapper.format(u';')
    COLON = _wrapper.format(u':')
    QUOTE = _wrapper.format(u'"')
    DASH = _wrapper.format(u'-')
    DOT = _wrapper.format(u'.')
    EXC_SIGN = _wrapper.format(u'!')
    QUESTION_SIGN = _wrapper.format(u'?')
    OPEN_BRACKET = _wrapper.format(u'OPEN_BRACKET')
    CLOSED_BRACKET = _wrapper.format(u'CLOSED_BRACKET')
    UNKNOWN_CHAR = _wrapper.format(u'UNKNOWN_CHAR')
    UNKNOWN_WORD = _wrapper.format(u'UNKNOWN_WORD')

    _token_mapping = {
        ',': COMMA,
        '.': DOT,
        '-': DASH,
        '?': QUESTION_SIGN,
        '!': EXC_SIGN,
        '(': OPEN_BRACKET,
        ')': CLOSED_BRACKET,
        '{': OPEN_BRACKET,
        '}': CLOSED_BRACKET,
        '[': OPEN_BRACKET,
        ']': CLOSED_BRACKET,
        '«': QUOTE,
        '»': QUOTE,
        '"': QUOTE,
    }

    @staticmethod
    def try_create(term):
        """
        Trying create a token by given 'term' parameter
        term: unicode
        return: unicode or None
            returns Token unicode or None
        """
        assert(isinstance(term, unicode))

        if term not in Tokens._token_mapping:
            return None
        return Tokens._token_mapping[term]


