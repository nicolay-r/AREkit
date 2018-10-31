# -*- coding: utf-8 -*-
from urlparse import urlparse


class Tokens:
    """
    Tokens used to describe a non-word text units, such as punctuation,
    uknown words/chars, smiles, etc.
    """

    _wrapper = u"<[{}]>"
    COMMA = _wrapper.format(u',')
    SEMICOLON = _wrapper.format(u';')
    COLON = _wrapper.format(u':')
    QUOTE = _wrapper.format(u'QUOTE')
    DASH = _wrapper.format(u'-')
    LONG_DASH = _wrapper.format(u'long_dash')
    DOT = _wrapper.format(u'.')
    TRIPLE_DOTS = _wrapper.format(u'…')
    EXC_SIGN = _wrapper.format(u'!')
    QUESTION_SIGN = _wrapper.format(u'?')
    OPEN_BRACKET = _wrapper.format(u'OPEN_BRACKET')
    CLOSED_BRACKET = _wrapper.format(u'CLOSED_BRACKET')
    NUMBER = _wrapper.format(u'NUMBER')
    NEW_LINE = _wrapper.format(u"NEW_LINE")
    UNKNOWN_CHAR = _wrapper.format(u'UNKNOWN_CHAR')
    UNKNOWN_WORD = _wrapper.format(u'UNKNOWN_WORD')
    URL = _wrapper.format(u"URL")

    _token_mapping = {
        u',': COMMA,
        u'.': DOT,
        u'…': TRIPLE_DOTS,
        u':': COLON,
        u';': SEMICOLON,
        u'-': DASH,
        u'—': LONG_DASH,
        u'?': QUESTION_SIGN,
        u'!': EXC_SIGN,
        u'(': OPEN_BRACKET,
        u')': CLOSED_BRACKET,
        u'{': OPEN_BRACKET,
        u'}': CLOSED_BRACKET,
        u'[': OPEN_BRACKET,
        u']': CLOSED_BRACKET,
        u'\n': NEW_LINE,
        u'«': QUOTE,
        u'»': QUOTE,
        u'"': QUOTE,
    }

    _supported_tokens = {
        COMMA,
        SEMICOLON,
        COLON,
        QUOTE,
        DASH,
        DOT,
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
        return: unicode or None
            returns Token unicode or None
        """
        assert(isinstance(subterm, unicode))

        if subterm not in Tokens._token_mapping:
            return None
        return Tokens._token_mapping[subterm]

    @staticmethod
    def try_create_number(term):
        assert(isinstance(term, unicode))
        return Tokens.NUMBER if term.isdigit() else None

    @staticmethod
    def try_create_url(term):
        assert(isinstance(term, unicode))
        result = urlparse(term)
        is_correct = result.scheme and result.netloc and result.path
        return Tokens.URL if is_correct else None

    @staticmethod
    def is_token(term):
        assert(isinstance(term, unicode))
        return term in Tokens._supported_tokens

    @staticmethod
    def iter_chars_by_token(term):
        """
        Iterate through charts that is related to term
        token: char
        """
        assert(isinstance(term, unicode))
        for char, token in Tokens._token_mapping.iteritems():
            if term == token:
                yield char
