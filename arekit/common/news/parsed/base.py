import collections

from arekit.common.entities.base import Entity
from arekit.common.news.parsed.term_position import TermPositionTypes, TermPosition
from arekit.common.text.parsed import BaseParsedText
from arekit.processing.text.enums import TermFormat


class ParsedNews(object):
    """
    This class represents an information of the processed news in following directions:
        - news words
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
        assert(isinstance(doc_id, int))
        assert(isinstance(parsed_sentences, collections.Iterable))

        self.__doc_id = doc_id
        self.__parsed_sentences = list(parsed_sentences)
        self.__entity_positions = None

        self.__init_entity_positions()

    # region properties

    @property
    def RelatedDocID(self):
        return self.__doc_id

    # endregion

    # region private methods

    @staticmethod
    def __is_entity(term):
        return isinstance(term, Entity)

    def __init_entity_positions(self):
        self.__entity_positions = self.__calculate_entity_positions()

    def __calculate_entity_positions(self):
        positions = {}
        t_ind_in_doc = 0

        for s_ind, t_ind_in_sent, term in self.__iter_all_raw_terms():

            if ParsedNews.__is_entity(term):
                positions[term.IdInDocument] = TermPosition(term_ind_in_doc=t_ind_in_doc,
                                                            term_ind_in_sent=t_ind_in_sent,
                                                            s_ind=s_ind)

            t_ind_in_doc += 1

        return positions

    def __iter_all_raw_terms(self, term_check=None, term_only=False):
        assert(callable(term_check) or term_check is None)
        assert(isinstance(term_only, bool))

        for s_ind, sentence in enumerate(self.__parsed_sentences):
            for ind_in_sent, term in self.__iter_sentence_raw_terms(sentence, term_check=term_check):

                if term_only:
                    yield term
                else:
                    yield s_ind, ind_in_sent, term

    @staticmethod
    def __iter_sentence_raw_terms(sentence, term_check):
        assert(isinstance(sentence, BaseParsedText))
        assert(callable(term_check) or term_check is None)

        for ind_in_sent, term in enumerate(sentence.iter_terms(TermFormat.Raw)):

            if term_check is not None:
                if not term_check(term):
                    continue

            yield ind_in_sent, term

    # endregion

    # region public 'get' methods

    def get_entity_position(self, id_in_document, position_type=None):
        """
        returns: TermPosition or int
        """
        assert(isinstance(position_type, TermPositionTypes) or position_type is None)

        e_pos = self.__entity_positions[id_in_document]
        assert(isinstance(e_pos, TermPosition))

        if position_type is None:
            return e_pos

        return e_pos.get_index(position_type)

    def get_entity_value(self, id_in_document):
        position = self.__entity_positions[id_in_document]
        assert(isinstance(position, TermPosition))
        sentence = self.__parsed_sentences[position.get_index(position_type=TermPositionTypes.SentenceIndex)]
        assert(isinstance(sentence, BaseParsedText))
        entity = sentence.get_term(position.get_index(position_type=TermPositionTypes.IndexInSentence),
                                   term_format=TermFormat.Raw)
        assert(isinstance(entity, Entity))
        return entity.Value

    # endregion

    # region public 'modify' methods

    def modify_parsed_sentences(self, sentence_objs_upd_func, get_obj_bound_func):
        assert(callable(sentence_objs_upd_func))
        assert(callable(get_obj_bound_func))

        for s_index, sentence in enumerate(self.__parsed_sentences):
            assert(isinstance(sentence, BaseParsedText))

            sentence.modify_by_bounded_objects(
                modified_objs=sentence_objs_upd_func(sentence),
                get_obj_bound_func=get_obj_bound_func)

        self.__init_entity_positions()

    # endregion

    # region public 'iter' methods

    def iter_terms(self, term_check=None):
        for term in self.__iter_all_raw_terms(term_only=True, term_check=term_check):
            yield term

    def iter_entities(self):
        return self.__iter_all_raw_terms(term_only=True, term_check=lambda term: self.__is_entity(term))

    def iter_sentence_terms(self, sentence_index, return_id, term_check=None):
        assert(isinstance(sentence_index, int))
        assert(isinstance(return_id, bool))
        assert(callable(term_check) or term_check is None)

        it = self.__iter_sentence_raw_terms(sentence=self.__parsed_sentences[sentence_index],
                                            term_check=term_check)

        for ind_in_sent, term in it:
            if return_id:
                yield ind_in_sent, term
            else:
                yield term

    def __iter__(self):
        for sentence in self.__parsed_sentences:
            yield sentence

    # endregion
