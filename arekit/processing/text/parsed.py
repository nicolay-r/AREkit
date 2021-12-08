from arekit.common.text.parsed import BaseParsedText
from arekit.common.text.stemmer import Stemmer
from arekit.processing.text.enums import TermFormat


class DefaultParsedText(BaseParsedText):
    """
    Represents a processed text with lemmatization support.
    """

    # region constructors

    def __init__(self, terms, stemmer=None):
        """
        NOTE: token hiding is actually discarded.
        """
        assert(isinstance(terms, list))
        assert(isinstance(stemmer, Stemmer) or stemmer is None)

        super(DefaultParsedText, self).__init__(terms=terms)

        self.__lemmas = None
        self.__stemmer = stemmer

        if stemmer is not None:
            self.__lemmatize(stemmer)

    def copy_modified(self, terms):
        return DefaultParsedText(terms=terms,
                                 stemmer=self.__stemmer)

    # endregion

    def _get_terms(self, term_format):
        return self.__lemmas if term_format == TermFormat.Lemma else \
            super(DefaultParsedText, self)._get_terms(term_format)

    # region private methods

    def __lemmatize(self, stemmer):
        """
        Compose a list of lemmatized versions of parsed_news
        PS: Might be significantly slow, depending on stemmer were used.
        """
        assert(isinstance(stemmer, Stemmer))
        self.__lemmas = ["".join(stemmer.lemmatize_to_list(term)) if isinstance(term, str) else term
                         for term in self._terms]

    # endregion
