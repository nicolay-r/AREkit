from core.common.entities.entity import Entity


class RuSentRelEntity(Entity):
    """ Annotated entity in RuSentRel corpus.
        Provides bounds, i.e. char indices in related sentence.
    """

    def __init__(self, doc_id, str_type, char_index_begin, char_index_end, value):
        assert(isinstance(doc_id, int))
        assert(isinstance(str_type, unicode))
        assert(isinstance(char_index_begin, int))
        assert(isinstance(char_index_end, int))
        super(RuSentRelEntity, self).__init__(value)

        self.__id = doc_id
        self.__str_type = str_type
        self.__begin = char_index_begin
        self.__end = char_index_end

    @property
    def CharIndexBegin(self):
        return self.__begin

    @property
    def CharIndexEnd(self):
        return self.__end

    @property
    def IdInDocument(self):
        return self.__id