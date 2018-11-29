# -*- coding: utf-8 -*-
from core.source.tokens import Tokens, Token
from core.processing.lemmatization.base import Stemmer


class TextParser:
    """
    Represents a parser of news sentences.
    Now uses in neural networks for text processing.
    As a result we have a list of terms, where term could be a
        1) Word
        2) Token
    """

    # TODO: Create unique name based on this template.
    _token_placeholder = "tokenplaceholder"

    def __init__(self):
        pass

    @staticmethod
    def parse(text, keep_tokens=False, stemmer=None, debug=False):
        terms = TextParser._parse_core(text, keep_tokens, stemmer, debug)
        return ParsedText(terms, hide_tokens=keep_tokens)

    @staticmethod
    def _parse_core(text, save_tokens=False, stemmer=None, debug=False):
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
        assert(isinstance(stemmer, Stemmer) or stemmer is None)

        # Separate everything via spaces.
        terms = [w.strip(u' ') for w in text.split(' ')]

        # Process with tokens
        processed_terms = TextParser._process_terms(terms, save_tokens)

        if stemmer is not None:

            # Temp replace tokens with placeholder
            tokens = []
            for i, t in enumerate(processed_terms):
                if isinstance(t, Token):
                    tokens.append(t)
                    processed_terms[i] = TextParser._token_placeholder

            # Pass to lemmatizer, and then check the result.
            processed_terms = stemmer.lemmatize_to_list(u' '.join(processed_terms))

            # Replace placeholder back
            j = 0
            for i, t in enumerate(processed_terms):
                if t == TextParser._token_placeholder and j < len(tokens):
                    processed_terms[i] = tokens[j]
                    j += 1

        if debug:
            TextParser._print(processed_terms)

        return processed_terms

    @staticmethod
    def _process_terms(terms, keep_tokens):
        """
        terms: list
            list of terms
        keep_tokes: bool
            keep or remove tokens from list of terms
        """
        assert(isinstance(terms, list))
        parsed = []
        for term in terms:

            if term is None:
                continue

            terms_with_tokens = TextParser._split_tokens(term)

            if not keep_tokens:
                terms_with_tokens = [term for term in terms_with_tokens if not isinstance(term, Token)]

            parsed.extend(terms_with_tokens)

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
            return [url]

        l = 0
        terms_with_tokens = []
        while l < len(term):
            # token
            token_value = Tokens.try_create(term[l])
            if token_value is not None:
                terms_with_tokens.append(Token(term[l], token_value))
                l += 1
            # number
            elif unicode.isdigit(term[l]):
                k = l + 1
                while k < len(term) and unicode.isdigit(term[k]):
                    k += 1
                token_value = Tokens.try_create_number(term[l:k])
                assert(token_value is not None)
                terms_with_tokens.append(Token(term[l:k], token_value))
                l = k
            # term
            else:
                k = l + 1
                while k < len(term):
                    token_value = Tokens.try_create(term[k])
                    if token_value is not None and token_value != Tokens.DASH:
                        break
                    k += 1
                terms_with_tokens.append(term[l:k])
                l = k

        return terms_with_tokens

    @staticmethod
    def _print(terms):
        for t in terms:
            print '"{}" '.format(t.encode('utf-8')),


class ParsedText:
    """
    Represents a processed text with extra parameters
    that were used during parsing.
    """

    _number_example = u"0"
    _url_example = u"http://sample.com"

    def __init__(self, terms, hide_tokens):
        assert(isinstance(terms, list))
        assert(isinstance(hide_tokens, bool))
        self._terms = terms
        self.token_values_hidden = hide_tokens

    def subtext(self, begin, end):
        assert(isinstance(begin, int))
        assert(isinstance(end, int))
        return ParsedText(self._terms[begin:end],
                          hide_tokens=self.token_values_hidden)

    @property
    def Terms(self):
        for term in self._terms:
            yield self._output_term(term)

    def get_term(self, i):
        return self._output_term(self._terms[i])

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

    def _get_token_as_term(self, token):
        return token.get_token_value() if self.token_values_hidden \
            else token.get_original_value()

    def __len__(self):
        return len(self._terms)

    def __iter__(self):
        for term in self._terms:
            yield term
