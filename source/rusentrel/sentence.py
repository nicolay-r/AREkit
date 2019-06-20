class RuSentRelSentence(object):
    """ Represent a raw sentence of rusentrel.
        Provides text could be used to parse then.
        Provides API to store entites.
    """

    # TODO. Remove begin / end parameters
    def __init__(self, text, begin, end):
        assert(isinstance(text, unicode) and len(text) > 0)
        assert(isinstance(begin, int))
        assert(isinstance(end, int))
        self.__text = text
        self.__begin = begin
        self.__end = end
        self.__entity_info = []

    # TODO. Remove
    @property
    def Begin(self):
        return self.__begin

    # TODO. Remove
    @property
    def End(self):
        return self.__end

    @property
    def Text(self):
        return self.__text

    # TODO. Entity as an argument instead of lots of params.
    def add_local_entity(self, id, begin, end):
        assert(isinstance(id, int))
        assert(isinstance(begin, int))
        assert(isinstance(end, int))
        self.__entity_info.append((id, begin, end))

    # TODO. Therefore simplify.
    def iter_entity_ids(self):
        for entity in self.__entity_info:
            yield entity[0]  # ID

    # TODO. iter by RuSentRelEntity objects
    def iter_entities_info(self):
        for info in self.__entity_info:
            yield info