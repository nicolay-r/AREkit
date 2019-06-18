class Sentence(object):

    def __init__(self, text, begin, end):
        assert(isinstance(text, unicode) and len(text) > 0)
        assert(isinstance(begin, int))
        assert(isinstance(end, int))
        self.__text = text
        self.__entity_info = []
        self.__begin = begin
        self.__end = end

    @property
    def Begin(self):
        return self.__begin

    @property
    def End(self):
        return self.__end

    @property
    def Text(self):
        return self.__text

    def add_local_entity(self, id, begin, end):
        assert(isinstance(id, int))
        assert(isinstance(begin, int))
        assert(isinstance(end, int))
        self.__entity_info.append((id, begin, end))

    def iter_entity_ids(self):
        for entity in self.__entity_info:
            yield entity[0]  # ID

    def iter_entities_info(self):
        for info in self.__entity_info:
            yield info