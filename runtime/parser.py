# -*- coding: utf-8 -*-
from core.source.tokens import Tokens, Token
from core.processing.lemmatization.base import Stemmer


class TextParser:
    """
    Represents a parser of news sentences.
    Now uses in neural networks for text processing.
    As a result we have a list of TERMS, where term could be a
        1) Word
        2) Token
    """

    # TODO: Create unique name based on this template.
    _token_placeholder = "tokenplaceholder"

    def __init__(self):
        pass

    @staticmethod
    def parse(text, keep_tokens=False, stemmer=None, debug=False):
        terms = TextParser._parse_core(text, keep_tokens, debug)
        return ParsedText(terms, hide_tokens=keep_tokens, stemmer=stemmer)

    @staticmethod
    def _parse_core(text, save_tokens=False, debug=False):
        """
        Separates sentence into list of terms

        save_tokens: bool
            keep token information in result list of terms.
        stemmer: None or Stemmer
            apply stemmer for lemmatization
        return: list
            list of unicode terms, where each term: word or token
        """
        assert(isinstance(text, unicode))
        assert(isinstance(save_tokens, bool))

        words = [word.strip(u' ') for word in text.split(' ')]
        terms = TextParser._process_words(words, save_tokens)

        if debug:
            TextParser._print(terms)

        return terms

    @staticmethod
    def _process_words(words, keep_tokens):
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

            words_and_tokens = TextParser._split_tokens(word)

            if not keep_tokens:
                words_and_tokens = [word for word in words_and_tokens if not isinstance(word, Token)]

            parsed.extend(words_and_tokens)

        return parsed


    @staticmethod
    def _split_tokens(term):
        """
        Splitting off tokens from terms ending, i.e. for example:
            term: "сказать,-" -> "(term: "сказать", ["COMMA_TOKEN", "DASH_TOKEN"])
        return: (unicode or None, list)
            modified term and list of extracted tokens.
        """

        url = Tokens.try_create_url(term)
        if url is not None:
            return [Token(url, Tokens.URL)]

        l = 0
        words_and_tokens = []
        while l < len(term):
            # token
            token_value = Tokens.try_create(term[l])
            if token_value is not None:
                words_and_tokens.append(Token(term[l], token_value))
                l += 1
            # number
            elif unicode.isdigit(term[l]):
                k = l + 1
                while k < len(term) and unicode.isdigit(term[k]):
                    k += 1
                token_value = Tokens.try_create_number(term[l:k])
                assert(token_value is not None)
                words_and_tokens.append(Token(term[l:k], token_value))
                l = k
            # term
            else:
                k = l + 1
                while k < len(term):
                    token_value = Tokens.try_create(term[k])
                    if token_value is not None and token_value != Tokens.DASH:
                        break
                    k += 1
                words_and_tokens.append(term[l:k])
                l = k

        return words_and_tokens

    @staticmethod
    def _print(terms):
        for term in terms:
            if isinstance(term, Token):
                print '"TOKEN: {}, {}" '.format(
                    term.get_original_value().encode('utf-8'),
                    term.get_token_value().encode('utf-8'))
            else:
                print '"WORD: {}" '.format(term.encode('utf-8'))


class ParsedText:
    """
    Represents a processed text with extra parameters
    that were used during parsing.
    """

    def __init__(self, terms, hide_tokens, stemmer=None):
        assert(isinstance(terms, list))
        assert(isinstance(hide_tokens, bool))
        assert(isinstance(stemmer, Stemmer) or stemmer is None)
        self._terms = terms
        self.token_values_hidden = hide_tokens
        self._lemmas = None
        if stemmer is not None:
            self._lemmatize(stemmer)

    def subtext(self, begin, end):
        assert(isinstance(begin, int))
        assert(isinstance(end, int))
        return ParsedText(self._terms[begin:end],
                          hide_tokens=self.token_values_hidden)

    @property
    # TODO: Processing outside. Method also might be renamed as 'iter_*'
    def Terms(self):
        for term in self._terms:
            yield self._output_term(term)

    def iter_raw_terms(self):
        for term in self._terms:
            yield term

    @property
    def Lemmas(self):
        assert(isinstance(self._lemmas, list))
        for lemma in self._lemmas:
            yield lemma

    def _lemmatize(self, stemmer):
        """
        Compose a list of lemmatized versions of terms
        PS: Might be significantly slow, depending on stemmer were used.
        """
        assert(isinstance(stemmer, Stemmer))
        self._lemmas = [self._get_token_as_term(t) if isinstance(t, Token)
                        else u"".join(stemmer.lemmatize_to_list(t))
                        for t in self._terms]

    def get_term(self, i):
        return self._output_term(self._terms[i])

    def get_lemma(self, i):
        return self._output_lemma(self._lemmas[i])

    def is_token_values_hidden(self):
        return self.token_values_hidden

    def unhide_token_values(self):
        """
        Display original token values, i.e. ',', '.'
        """
        self.token_values_hidden = False

    def hide_token_values(self):
        """
        Display tokens as '<[COMMA]>', etc.
        """
        self.token_values_hidden = True

    def _output_term(self, term):
        return self._get_token_as_term(term) if isinstance(term, Token) else term

    def _output_lemma(self, lemma):
        return self._get_token_as_term(lemma) if isinstance(lemma, Token) else lemma

    def _get_token_as_term(self, token):
        return token.get_token_value() if self.token_values_hidden \
            else token.get_original_value()

    def __len__(self):
        return len(self._terms)

    def __iter__(self):
        for term in self._terms:
            yield term
