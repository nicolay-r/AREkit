from core.common.entities.entity import Entity


class RuSentRelEntity(Entity):
    """ Entity description.
    """

    def __init__(self, doc_id, str_type, begin, end, value):
        assert(type(doc_id) == unicode)
        assert(type(str_type) == unicode)
        assert(type(begin) == int)
        assert(type(end) == int)
        super(RuSentRelEntity, self).__init__(value)

        self.__doc_id = doc_id
        self.__str_type = str_type
        self.__begin = begin
        self.__end = end

    @property
    def Begin(self):
        return self.__begin

    @property
    def IdInDocument(self):
        return int(self.__doc_id[1:len(self.__doc_id)])