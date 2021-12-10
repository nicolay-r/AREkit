import collections

from arekit.processing.text.enums import TermFormat


class BaseParsedText(object):
    """
    Represents a processed text with extra parameters
    that were used during parsing.
    """

    # region constructors

    def __init__(self, terms):
        """
        NOTE: token hiding is actually discarded.
        """
        assert(isinstance(terms, list))
        self._terms = terms

    def copy_modified(self, terms):
        raise NotImplementedError()

    # endregion

    def modify_by_bounded_objects(self, modified_objs, get_obj_bound_func):
        assert(isinstance(modified_objs, collections.Iterable))
        assert(callable(get_obj_bound_func))

        def __remove(terms, start, end):
            while end > start:
                del terms[start]
                end -= 1

        if modified_objs is None:
            return

        objs_list = list(modified_objs)

        # setup default position.
        local_terms = self._terms
        prev_position = len(self._terms) + 1

        for obj in reversed(objs_list):
            obj_bound = get_obj_bound_func(obj)

            if obj_bound.Position > prev_position:
                raise Exception("objs list has incorrect order. It is expected that instances "
                                "ordered by positions (ascending)")

            __remove(terms=local_terms,
                     start=obj_bound.Position,
                     end=obj_bound.Position + obj_bound.Length)

            local_terms.insert(obj_bound.Position, obj)

            prev_position = obj_bound.Position

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