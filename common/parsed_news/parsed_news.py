from core.common.entities.entity import Entity


class ParsedNews(object):
    """
    Extracted News lexemes, such as:
        - news words
        - tokens
        - entities (positions).
    """

    def __init__(self, news_id, terms, sentence_begin_inds):
        """
        news_id:
        terms:
        entity_positions: list of pairs (document term index, document sentence index)
        sentences_count:
        sentence_begin_inds:
        """
        assert(isinstance(news_id, int))
        assert(isinstance(terms, list))
        assert(isinstance(sentence_begin_inds, list))
        self.__news_id = news_id
        self.__terms = terms
        self.__sentence_begin_inds = sentence_begin_inds
        self.__sentences_count = len(sentence_begin_inds)
        self.__entity_positions = self.__init_positions()

    @property
    def RelatedNewsID(self):
        return self.__news_id

    def __init_positions(self):
        positions = {}
        for term_index, term in enumerate(self.__terms):
            if not isinstance(term, Entity):
                continue
            positions[term.IdInDocument] = (term_index, self.__get_sentence_ind(term_index))
        return positions

    def __get_sentence_ind(self, term_index):
        for s_index, start in enumerate(self.__sentence_begin_inds):
            if start == term_index:
                return s_index
            elif start > term_index:
                return s_index - 1

    def get_term_index_in_sentence(self, term_index):
        assert(isinstance(term_index, int))
        begin = 0
        for i, begin_index in enumerate(self.__sentence_begin_inds):
            if begin_index > term_index:
                break
            begin = begin_index

        return term_index - begin

    def get_entity_term_index(self, id_in_document):
        pair = self.__entity_positions[id_in_document]
        return pair[0]

    def get_entity_sentence_index(self, id_in_document):
        pair = self.__entity_positions[id_in_document]
        return pair[1]

    def get_entity_value(self, id_in_document):
        t_ind = self.get_entity_term_index(id_in_document)
        entity = self.__terms[t_ind]
        assert(isinstance(entity, Entity))
        return entity.Value

    def iter_terms(self):
        for term in self.__terms:
            yield term

    def iter_sentence_terms(self, sentence_index):
        assert(isinstance(sentence_index, int))
        begin = self.__sentence_begin_inds[sentence_index]
        end = len(self.__terms) if sentence_index == self.__sentences_count - 1 \
            else self.__sentence_begin_inds[sentence_index + 1]
        for i in range(begin, end):
            yield self.__terms[i]

    def __len__(self):
        return len(self.__terms)

