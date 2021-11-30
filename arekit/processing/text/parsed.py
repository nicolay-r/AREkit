import collections

from arekit.processing.lemmatization.base import Stemmer
from arekit.processing.text.enums import TermFormat


# TODO. Move it as a BaseParsedText into common (leave only terms)
# TODO. LEMMATIZATION leave within this class
# TODO. By default we may not need to perform lemmatization (BaseParsedText lacks of lemmatization)
class ParsedText:
    """
    Represents a processed text with extra parameters
    that were used during parsing.
    """

    # region constructors

    # TODO. move to base (keep terms only)
    def __init__(self, terms, stemmer=None):
        """
        NOTE: token hiding is actually discarded.
        """
        assert(isinstance(terms, list))
        # TODO. leave here.
        assert(isinstance(stemmer, Stemmer) or stemmer is None)

        self.__terms = terms
        # TODO. leave here.
        self.__lemmas = None
        # TODO. leave here.
        self.__stemmer = stemmer

        self.__update_lemmatization()

    def __update_lemmatization(self):
        # TODO. leave here.
        if self.__stemmer is not None:
            self.__lemmatize(self.__stemmer)

    # TODO. to base as an abstract-like method. (implementation here)
    def copy_modified(self, terms):
        return ParsedText(terms=terms,
                          stemmer=self.__stemmer)

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
        prev_position = len(objs_list) + 1
        local_terms = self.__terms

        for obj in reversed(objs_list):
            obj_bound = get_obj_bound_func(obj)

            if obj_bound.Position > prev_position:
                raise Exception("objs list has incorrect order. It is expected that instances "
                                "ordered by positions (ascending)")

            __remove(terms=local_terms,
                     start=obj_bound.Position,
                     end=obj_bound.Position + obj_bound.Length)

            local_terms.insert(obj_bound.Position, obj)

        self.__update_lemmatization()

    # endregion

    # TODO. move to base.
    def get_term(self, index, term_format):
        assert(isinstance(term_format, TermFormat))
        return self.__src(term_format)[index]

    # TODO. move to base.
    def iter_terms(self, term_format, filter=None):
        assert(isinstance(term_format, TermFormat))
        assert(callable(filter) or filter is None)
        src = self.__src(term_format)
        for term in src:
            if filter is not None and not list(filter(term)):
                continue
            yield term

    # region private methods

    # TODO. move to base.
    def __src(self, term_format):
        assert(isinstance(term_format, TermFormat))
        return self.__lemmas if term_format == term_format.Lemma else self.__terms

    def __lemmatize(self, stemmer):
        """
        Compose a list of lemmatized versions of parsed_news
        PS: Might be significantly slow, depending on stemmer were used.
        """
        assert(isinstance(stemmer, Stemmer))
        self.__lemmas = ["".join(stemmer.lemmatize_to_list(t)) if isinstance(t, str) else t
                         for t in self.__terms]

    # endregion

    # TODO. move to base.
    def __len__(self):
        return len(self.__terms)
