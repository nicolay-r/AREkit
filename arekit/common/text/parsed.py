from arekit.common.text.enums import TermFormat


class BaseParsedText(object):
    """
    Represents a processed text with extra parameters
    that were used during parsing.
    """

    # region constructors

    def __init__(self, terms):
        assert(isinstance(terms, list))
        self._terms = terms

    # endregion

    def get_term(self, index, term_format):
        assert(isinstance(term_format, TermFormat))
        terms = self._get_terms(term_format)
        return terms[index]

    def iter_terms(self, term_format, filter=None):
        assert(isinstance(term_format, TermFormat))
        assert(callable(filter) or filter is None)
        terms = self._get_terms(term_format)
        for term in terms:
            if filter is not None and not list(filter(term)):
                continue
            yield term

    # region private methods

    def _get_terms(self, term_format):
        assert(isinstance(term_format, TermFormat))
        assert(term_format == TermFormat.Raw)
        return self._terms

    # endregion

    def __len__(self):
        return len(self._terms)