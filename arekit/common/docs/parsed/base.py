from collections.abc import Iterable

from arekit.common.entities.base import Entity
from arekit.common.text.enums import TermFormat
from arekit.common.text.parsed import BaseParsedText


class ParsedDocument(object):
    """
    This class represents an information of the processed doc in following directions:
        - doc words
        - tokens
        - entities (positions).
        - frames (FrameVariants)
    It allows:
        - Expand parsed sentences with other objects:
            modify_parsed_sentences(func)

    Limitations:
        IN MEMORY implementation (`add` method)
    """

    def __init__(self, doc_id, parsed_sentences):
        """
        parsed_sentences: iterable of ParsedSentence type
            NOTE: Considered sentences with labeled Entities in it!
        """
        assert(isinstance(parsed_sentences, Iterable))

        self.__doc_id = doc_id
        self.__parsed_sentences = list(parsed_sentences)

    # region properties

    @property
    def RelatedDocID(self):
        return self.__doc_id

    # endregion

    # region private methods

    def __iter_all_raw_terms(self, filter_func=None, term_only=False):
        assert(callable(filter_func) or filter_func is None)
        assert(isinstance(term_only, bool))

        for s_ind, sentence in enumerate(self.__parsed_sentences):
            for ind_in_sent, term in self.__iter_sentence_raw_terms(sentence, filter_func=filter_func):

                if term_only:
                    yield term
                else:
                    yield s_ind, ind_in_sent, term

    @staticmethod
    def __iter_sentence_raw_terms(sentence, filter_func):
        assert(isinstance(sentence, BaseParsedText))
        assert(callable(filter_func) or filter_func is None)

        for ind_in_sent, term in enumerate(sentence.iter_terms(TermFormat.Raw)):

            if filter_func is not None:
                if not filter_func(term):
                    continue

            yield ind_in_sent, term

    # endregion

    # region public 'iter' methods

    def get_sentence(self, s_ind):
        assert(isinstance(s_ind, int))
        return self.__parsed_sentences[s_ind]

    def iter_entities(self):
        for entity in self.__iter_all_raw_terms(term_only=True, filter_func=lambda t: isinstance(t, Entity)):
            yield entity

    def iter_terms(self, filter_func=None, term_only=True):
        for term in self.__iter_all_raw_terms(term_only=term_only, filter_func=filter_func):
            yield term

    def iter_sentence_terms(self, sentence_index, return_id, filter_func=None):
        assert(isinstance(sentence_index, int))
        assert(isinstance(return_id, bool))
        assert(callable(filter_func) or filter_func is None)

        it = self.__iter_sentence_raw_terms(sentence=self.__parsed_sentences[sentence_index],
                                            filter_func=filter_func)

        for ind_in_sent, term in it:
            if return_id:
                yield ind_in_sent, term
            else:
                yield term
    # endregion

    def __iter__(self):
        for sentence in self.__parsed_sentences:
            yield sentence
