# -*- coding: utf-8 -*-
from core.source.tokens import Tokens, Token
from core.processing.lemmatization.base import Stemmer


# TODO. Move into processing/text directory.
class TextParser:
    """
    Represents a parser of news sentences.
    Now uses in neural networks for text processing.
    As a result we have a list of TERMS, where term could be a
        1) Word
        2) Token
    """

    def __init__(self):
        pass

    @staticmethod
    def parse(text, keep_tokens=False, stemmer=None, debug=False):
        terms = TextParser.__parse_core(text, keep_tokens, debug=debug)
        return ParsedText(terms, hide_tokens=keep_tokens, stemmer=stemmer)

    @staticmethod
    def from_string(str, separator=' ', keep_tokens=True, stemmer=None):

        def __term_or_token(term):
            token = TextParser.__try_term_as_token(term)
            return token if token is not None else term

        assert(isinstance(str, str))
        terms = [word.strip(' ') for word in str.split(separator)]
        terms = [__term_or_token(t) for t in terms]
        return ParsedText(terms, hide_tokens=keep_tokens, stemmer=stemmer)

    @staticmethod
    def __parse_core(text, keep_tokens=False, debug=False):
        """
        Separates sentence into list of terms

        save_tokens: bool
            keep token information in result list of terms.
        return: list
            list of unicode terms, where each term: word or token
        """
        assert(isinstance(text, str))
        assert(isinstance(keep_tokens, bool))

        words = [word.strip(' ') for word in text.split(' ')]
        terms = TextParser.__process_words(words, keep_tokens)

        if debug:
            TextParser.__print(terms)

        return terms

    @staticmethod
    def __process_words(words, keep_tokens):
        """
        terms: list
            list of terms
        keep_tokes: bool
            keep or remove tokens from list of terms
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
        Splitting off tokens from terms ending, i.e. for example:
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
            elif str.isdigit(term[l]):
                k = l + 1
                while k < len(term) and str.isdigit(term[k]):
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
                print('"TOKEN: {}, {}" '.format(
                    term.get_original_value(),
                    term.get_token_value()).decode('utf-8'))
            else:
                print('"WORD: {}" '.format(term).decode('utf-8'))

# TODO. Move into processing/text directory.
class ParsedText:
    """
    Represents a processed text with extra parameters
    that were used during parsing.
    """

    def __init__(self, terms, hide_tokens, stemmer=None):
        assert(isinstance(terms, list))
        assert(isinstance(hide_tokens, bool))
        assert(isinstance(stemmer, Stemmer) or stemmer is None)
        self.__terms = terms
        self.__hide_token_value = hide_tokens
        self.__lemmas = None
        if stemmer is not None:
            self.__lemmatize(stemmer)

    @property
    def IsTokenValuesHidden(self):
        return self.__hide_token_value

    @property
    # TODO: Processing outside. Method also might be renamed as 'iter_*'
    def Terms(self):
        for term in self.__terms:
            yield self.__output_term(term, self.hide_token_values())

    def is_term(self, index):
        assert(isinstance(index, int))
        return isinstance(self.__terms[index], str)

    def iter_lemmas(self):
        for lemma in self.__lemmas:
            yield self.__output_term(lemma, self.hide_token_values())

    def iter_raw_terms(self):
        for term in self.__terms:
            yield term

    def iter_raw_words(self):
        for term in self.__terms:
            if not isinstance(term, str):
                continue
            yield term

    def iter_raw_lemmas(self):
        for lemma in self.__lemmas:
            yield lemma

    def iter_raw_word_lemmas(self):
        for lemma in self.__lemmas:
            if not isinstance(lemma, str):
                continue
            yield lemma

    def __lemmatize(self, stemmer):
        """
        Compose a list of lemmatized versions of terms
        PS: Might be significantly slow, depending on stemmer were used.
        """
        assert(isinstance(stemmer, Stemmer))
        self.__lemmas = ["".join(stemmer.lemmatize_to_list(t)) if isinstance(t, str) else t
                         for t in self.__terms]

    def get_term(self, i):
        return self.__output_term(self.__terms[i], self.__hide_token_value)

    def get_lemma(self, i):
        return self.__output_term(self.__terms[i], self.__hide_token_value)

    def is_token_values_hidden(self):
        return self.__hide_token_value

    def unhide_token_values(self):
        """
        Display original token values, i.e. ',', '.'
        """
        self.__hide_token_value = False

    def hide_token_values(self):
        """
        Display tokens as '<[COMMA]>', etc.
        """
        self.__hide_token_value = True

    def to_string(self):
        terms = [ParsedText.__output_term(term, False) for term in self.__terms]
        return ' '.join(terms)

    @staticmethod
    def __output_term(term, hide_token_value):
        return ParsedText.__get_token_as_term(term, hide_token_value) if isinstance(term, Token) else term

    @staticmethod
    def __get_token_as_term(token, hide):
        return token.get_token_value() if hide else token.get_original_value()

    def __len__(self):
        return len(self.__terms)

    def __iter__(self):
        for term in self.__terms:
            yield term
