from arekit.common.entities.base import Entity


class RuSentRelEntity(Entity):
    """ Annotated entity in RuSentRel corpus.
        Provides bounds, i.e. char indices in related sentence.
    """

    def __init__(self, id_in_doc, e_type, char_index_begin, char_index_end, value):
        assert(isinstance(e_type, str))
        assert(isinstance(char_index_begin, int))
        assert(isinstance(char_index_end, int))
        super(RuSentRelEntity, self).__init__(value=value, e_type=e_type)

        self.__e_type = e_type
        self.__begin = char_index_begin
        self.__end = char_index_end
        self.__id = id_in_doc

    @property
    def CharIndexBegin(self):
        return self.__begin

    @property
    def CharIndexEnd(self):
        return self.__end

    @property
    def Type(self):
        return self.__e_type

    @property
    def ID(self):
        return self.__id