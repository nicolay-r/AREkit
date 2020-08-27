from arekit.processing.lemmatization.base import Stemmer
from arekit.processing.text.enums import TermFormat


class ParsedText:
    """
    Represents a processed text with extra parameters
    that were used during parsing.
    """

    # region constructors

    def __init__(self, terms, stemmer=None):
        """
        NOTE: token hiding is actually discarded.
        """
        assert(isinstance(terms, list))
        assert(isinstance(stemmer, Stemmer) or stemmer is None)

        self.__terms = terms
        self.__lemmas = None
        self.__stemmer = stemmer

        if stemmer is not None:
            self.__lemmatize(stemmer)

    def copy_modified(self, terms):
        return ParsedText(terms=terms,
                          stemmer=self.__stemmer)

    # endregion

    def get_term(self, index, term_format):
        assert(isinstance(term_format, TermFormat))
        return self.__src(term_format)[index]

    def iter_terms(self, term_format, filter=None):
        assert(isinstance(term_format, TermFormat))
        assert(callable(filter) or filter is None)
        src = self.__src(term_format)
        for term in src:
            if filter is not None and not filter(term):
                continue
            yield term

    # region private methods

    def __src(self, term_format):
        assert(isinstance(term_format, TermFormat))
        return self.__lemmas if term_format == term_format.Lemma else self.__terms

    def __lemmatize(self, stemmer):
        """
        Compose a list of lemmatized versions of parsed_news
        PS: Might be significantly slow, depending on stemmer were used.
        """
        assert(isinstance(stemmer, Stemmer))
        self.__lemmas = [u"".join(stemmer.lemmatize_to_list(t)) if isinstance(t, unicode) else t
                         for t in self.__terms]

    # endregion

    def __len__(self):
        return len(self.__terms)
