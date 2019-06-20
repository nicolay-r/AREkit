from core.source.rusentrel.entities.entity import RuSentRelEntity


class RuSentRelSentence(object):
    """ Represent a raw sentence of rusentrel.
        Provides text could be used to parse then.
        Provides API to store entites.
    """

    # TODO. Remove begin / end parameters
    def __init__(self, text, char_ind_begin, char_ind_end):
        assert(isinstance(text, unicode) and len(text) > 0)
        assert(isinstance(char_ind_begin, int))
        assert(isinstance(char_ind_end, int))
        self.__text = text
        self.__begin = char_ind_begin
        self.__end = char_ind_end
        self.__entities = []

    @property
    def Text(self):
        return self.__text

    def add_local_entity(self, entity):
        assert(isinstance(entity, RuSentRelEntity))
        self.__entities.append(entity)

    def iter_entity_ids(self):
        for entity in self.__entities:
            yield entity.IdInDocument

    def iter_entities(self):
        for entity in self.__entities:
            yield entity

    def is_entity_goes_after(self, entity):
        assert(isinstance(entity, RuSentRelEntity))
        return entity.CharIndexBegin > self.__end

    def __contains__(self, entity):
        assert(isinstance(entity, RuSentRelEntity))
        return entity.CharIndexBegin >= self.__begin and entity.CharIndexEnd <= self.__end
