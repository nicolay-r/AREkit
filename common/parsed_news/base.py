import collections

from core.common.entities.base import Entity
from core.common.parsed_news.term_position import TermPosition
from core.common.text_frame_variant import TextFrameVariant
from core.processing.text.parsed import ParsedText


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
        - Modify entity type values as follows:
            modify_entity_types(func)
    """

    def __init__(self, news_id, parsed_sentences):
        """
        parsed_sentences: iterable of ParsedSentence type
            NOTE: Considered sentences with labeled Entities in it!
        """
        assert(isinstance(news_id, int))
        assert(isinstance(parsed_sentences, collections.Iterable))

        self.__news_id = news_id
        self.__parsed_sentences = list(parsed_sentences)
        self.__entity_positions = self.__init_positions()

    # region properties

    @property
    def RelatedNewsID(self):
        return self.__news_id

    # endregion

    # region private methods

    def __init_positions(self):
        positions = {}
        dl_term_index = 0

        for s_index, sentence in enumerate(self.__parsed_sentences):
            for sl_term_index, term in enumerate(sentence.iter_raw_terms()):

                if isinstance(term, Entity):
                    positions[term.IdInDocument] = TermPosition(doc_level_ind=dl_term_index,
                                                                s_level_ind=sl_term_index,
                                                                s_ind=s_index)
                dl_term_index += 1

        return positions

    def __iter_all_entities(self):
        for sentence in self.__parsed_sentences:
            for term in sentence.iter_raw_terms():
                if isinstance(term, Entity):
                    yield term

    # endregion

    # region public 'get' methods

    def get_entity_sentence_level_term_index(self, id_in_document):
        position = self.__entity_positions[id_in_document]
        return position.SentenceLevelIndex

    def get_entity_document_level_term_index(self, id_in_document):
        position = self.__entity_positions[id_in_document]
        return position.DocLevelIndex

    def get_entity_sentence_index(self, id_in_document):
        position = self.__entity_positions[id_in_document]
        return position.SentenceIndex

    def get_entity_value(self, id_in_document):
        position = self.__entity_positions[id_in_document]
        assert(isinstance(position, TermPosition))
        sentence = self.__parsed_sentences[position.SentenceIndex]
        assert(isinstance(sentence, ParsedText))
        entity = sentence.get_term(position.SentenceLevelIndex)
        assert(isinstance(entity, Entity))
        return entity.Value

    # endregion

    # region public 'modify' methods

    def modify_parsed_sentences(self, sentence_upd_func):
        assert(callable(sentence_upd_func))

        for s_index, sentence in enumerate(self.__parsed_sentences):
            updated = sentence_upd_func(sentence)
            assert(isinstance(updated, ParsedText))
            self.__parsed_sentences[s_index] = updated

        self.__entity_positions = self.__init_positions()

    def modify_entity_types(self, value_to_type_func):
        assert(callable(value_to_type_func))
        for e in self.__iter_all_entities():

            value = value_to_type_func(e.Value)

            if value is None:
                continue

            e.modify_type(value)

    # endregion

    # region public 'iter' methods

    def iter_terms(self):
        for sentence in self.__parsed_sentences:
            assert(isinstance(sentence, ParsedText))
            for term in sentence.iter_raw_terms():
                yield term

    def iter_sentence_frame_variants_with_indices(self, sentence_index):
        assert(isinstance(sentence_index, int))
        sentence = self.__parsed_sentences[sentence_index]
        assert(isinstance(sentence, ParsedText))
        for index, term in enumerate(sentence.iter_raw_terms()):
            if isinstance(term, TextFrameVariant):
                yield index, term

    def iter_sentence_entities_with_indices(self, sentence_index):
        assert(isinstance(sentence_index, int))
        sentence = self.__parsed_sentences[sentence_index]
        assert(isinstance(sentence, ParsedText))
        for index, term in enumerate(sentence.iter_raw_terms()):
            if isinstance(term, Entity):
                yield index, term

    def iter_sentence_terms(self, sentence_index):
        assert(isinstance(sentence_index, int))
        sentence = self.__parsed_sentences[sentence_index]
        assert(isinstance(sentence, ParsedText))
        for term in sentence.iter_raw_terms():
            yield term

    def __iter__(self):
        for sentence in self.__parsed_sentences:
            yield sentence

    # endregion
