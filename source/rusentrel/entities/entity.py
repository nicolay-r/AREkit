from core.common.entities.entity import Entity


class RuSentRelEntity(Entity):
    """ Entity description.
    """

    def __init__(self, doc_id, str_type, begin, end, value):
        assert(isinstance(doc_id, int))
        assert(isinstance(str_type, unicode))
        assert(isinstance(begin, int))
        assert(isinstance(end, int))
        super(RuSentRelEntity, self).__init__(value)

        self.__id = doc_id
        self.__str_type = str_type
        self.__begin = begin
        self.__end = end

    @property
    def Begin(self):
        return self.__begin

    @property
    def End(self):
        return self.__end

    @property
    def IdInDocument(self):
        return self.__id