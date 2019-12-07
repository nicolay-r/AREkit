from arekit.processing.lemmatization.base import Stemmer
from arekit.processing.text.token import Token


class ParsedText:
    """
    Represents a processed text with extra parameters
    that were used during parsing.
    """

    # region constructors

    def __init__(self, terms, hide_tokens, stemmer=None):
        assert(isinstance(terms, list))
        assert(isinstance(hide_tokens, bool))
        assert(isinstance(stemmer, Stemmer) or stemmer is None)
        self.__terms = terms
        self.__hide_token_value = hide_tokens
        self.__lemmas = None
        self.__stemmer = stemmer
        if stemmer is not None:
            self.__lemmatize(stemmer)

    def copy_modified(self, terms):
        return ParsedText(terms=terms,
                          hide_tokens=self.__hide_token_value,
                          stemmer=self.__stemmer)

    # endregion

    # region properties

    @property
    def IsTokenValuesHidden(self):
        return self.__hide_token_value

    @property
    # TODO: Processing outside. Method also might be renamed as 'iter_*'
    def Terms(self):
        for term in self.__terms:
            yield self.__output_term(term, self.hide_token_values())

    # endregion

    def is_term(self, index):
        assert(isinstance(index, int))
        return isinstance(self.__terms[index], unicode)

    # region 'iter' methods

    def iter_lemmas(self):
        for lemma in self.__lemmas:
            yield self.__output_term(lemma, self.hide_token_values())

    def iter_raw_terms(self):
        for term in self.__terms:
            yield term

    def iter_raw_words(self):
        for term in self.__terms:
            if not isinstance(term, unicode):
                continue
            yield term

    def iter_raw_lemmas(self):
        for lemma in self.__lemmas:
            yield lemma

    def iter_raw_word_lemmas(self):
        for lemma in self.__lemmas:
            if not isinstance(lemma, unicode):
                continue
            yield lemma

    # endregion

    def __lemmatize(self, stemmer):
        """
        Compose a list of lemmatized versions of parsed_news
        PS: Might be significantly slow, depending on stemmer were used.
        """
        assert(isinstance(stemmer, Stemmer))
        self.__lemmas = [u"".join(stemmer.lemmatize_to_list(t)) if isinstance(t, unicode) else t
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
        return u' '.join(terms)

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