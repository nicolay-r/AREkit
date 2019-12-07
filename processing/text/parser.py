# -*- coding: utf-8 -*-
import collections

from arekit.processing.text.parsed import ParsedText
from arekit.processing.text.tokens import Tokens
from arekit.processing.text.token import Token


class TextParser:
    """
    Represents a parser of news sentences.
    As a result we have a list of TERMS, where term could be a
        1) Word
        2) Token
    """

    def __init__(self):
        pass

    @staticmethod
    def parse(text, keep_tokens=False, stemmer=None, debug=False):
        assert(isinstance(text, unicode))
        terms = TextParser.__parse_core(text, keep_tokens, debug=debug)
        return ParsedText(terms, hide_tokens=keep_tokens, stemmer=stemmer)

    @staticmethod
    def parse_string_list(string_iter, keep_tokens=False, stemmer=None, debug=False):
        assert(isinstance(string_iter, collections.Iterable))

        terms = []
        for text in string_iter:
            if not isinstance(text, unicode):
                terms.append(text)
                continue
            new_terms = TextParser.__parse_core(text, keep_tokens, debug=debug)
            terms.extend(new_terms)

        return ParsedText(terms, hide_tokens=keep_tokens, stemmer=stemmer)

    @staticmethod
    def from_string(str, separator=u' ', keep_tokens=True, stemmer=None):

        def __term_or_token(term):
            token = TextParser.__try_term_as_token(term)
            return token if token is not None else term

        assert(isinstance(str, unicode))
        terms = [word.strip(u' ') for word in str.split(separator)]
        terms = [__term_or_token(t) for t in terms]
        return ParsedText(terms, hide_tokens=keep_tokens, stemmer=stemmer)

    @staticmethod
    def __parse_core(text, keep_tokens=False, debug=False):
        """
        Separates sentence into list of parsed_news

        save_tokens: bool
            keep token information in result list of parsed_news.
        return: list
            list of unicode parsed_news, where each term: word or token
        """
        assert(isinstance(text, unicode))
        assert(isinstance(keep_tokens, bool))

        words = [word.strip(u' ') for word in text.split(u' ')]
        terms = TextParser.__process_words(words, keep_tokens)

        if debug:
            TextParser.__print(terms)

        return terms

    @staticmethod
    def __process_words(words, keep_tokens):
        """
        parsed_news: list
            list of parsed_news
        keep_tokes: bool
            keep or remove tokens from list of parsed_news
        """
        assert(isinstance(words, list))
        parsed = []
        for word in words:

            if word is None:
                continue

            words_and_tokens = TextParser.__split_tokens(word)

            if not keep_tokens:
                words_and_tokens = [word for word in words_and_tokens if not isinstance(word, Token)]

            parsed.extend(words_and_tokens)

        return parsed

    @staticmethod
    def __split_tokens(term):
        """
        Splitting off tokens from parsed_news ending, i.e. for example:
            term: "сказать,-" -> "(term: "сказать", ["COMMA_TOKEN", "DASH_TOKEN"])
        return: (unicode or None, list)
            modified term and list of extracted tokens.
        """

        url = Tokens.try_create_url(term)
        if url is not None:
            return [url]

        l = 0
        words_and_tokens = []
        while l < len(term):
            # token
            token = Tokens.try_create(term[l])
            if token is not None:
                if token.get_token_value() != Tokens.NEW_LINE:
                    words_and_tokens.append(token)
                l += 1
            # number
            elif unicode.isdigit(term[l]):
                k = l + 1
                while k < len(term) and unicode.isdigit(term[k]):
                    k += 1
                token = Tokens.try_create_number(term[l:k])
                assert(token is not None)
                words_and_tokens.append(token)
                l = k
            # term
            else:
                k = l + 1
                while k < len(term):
                    token = Tokens.try_create(term[k])
                    if token is not None and token.get_token_value() != Tokens.DASH:
                        break
                    k += 1
                words_and_tokens.append(term[l:k])
                l = k

        return words_and_tokens

    @staticmethod
    def __try_term_as_token(term):
        url = Tokens.try_create_url(term)
        if url is not None:
            return url
        number = Tokens.try_create_number(term)
        if number is not None:
            return number
        return Tokens.try_create(term)

    @staticmethod
    def __print(terms):
        for term in terms:
            if isinstance(term, Token):
                print u'"TOKEN: {}, {}" '.format(
                    term.get_original_value(),
                    term.get_token_value()).decode('utf-8')
            else:
                print u'"WORD: {}" '.format(term).decode('utf-8')

