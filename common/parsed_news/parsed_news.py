import collections

from core.common.entities.entity import Entity
from core.common.parsed_news.term_position import TermPosition
from core.processing.text.parsed import ParsedText


# TODO: Adding frames task
# TODO: Provide api that allows to obtain frames (directly or
# TODO: via another way using iterators by a contents of news sentences)
# TODO: iter_sentence_term
from core.source.rusentiframes.variants.text_variant import TextFrameVariant


class ParsedNews(object):
    """
    Extracted News lexemes, such as:
        - news words
        - tokens
        - entities (positions).
    Allow to expand parsed sentences with other objects:
        modify_parsed_sentences(func)
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

    # endregion

    # region public methods

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

    def modify_parsed_sentences(self, sentence_upd_func):
        assert(callable(sentence_upd_func))

        for s_index, sentence in enumerate(self.__parsed_sentences):
            updated = sentence_upd_func(sentence)
            assert(isinstance(updated, ParsedText))
            self.__parsed_sentences[s_index] = updated

        self.__entity_positions = self.__init_positions()

    def iter_terms(self):
        for sentence in self.__parsed_sentences:
            assert(isinstance(sentence, ParsedText))
            for term in sentence.iter_raw_terms():
                yield term

    def iter_sentence_frame_indices(self, sentence_index):
        assert(isinstance(sentence_index, int))
        sentence = self.__parsed_sentences[sentence_index]
        assert(isinstance(sentence, ParsedText))
        for index, term in enumerate(sentence.iter_raw_terms()):
            if isinstance(term, TextFrameVariant):
                yield index

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
