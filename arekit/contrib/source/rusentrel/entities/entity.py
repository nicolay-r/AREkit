from arekit.common.entities.base import Entity


class RuSentRelEntity(Entity):
    """ Annotated entity in RuSentRel corpus.
        Provides bounds, i.e. char indices in related sentence.
    """

    def __init__(self, id_in_doc, e_type, char_index_begin, char_index_end, value):
        assert(isinstance(id_in_doc, int))
        assert(isinstance(e_type, unicode))
        assert(isinstance(char_index_begin, int))
        assert(isinstance(char_index_end, int))
        super(RuSentRelEntity, self).__init__(value=value,
                                              e_type=e_type,
                                              id_in_doc=id_in_doc)

        self.__e_type = e_type
        self.__begin = char_index_begin
        self.__end = char_index_end

    @property
    def CharIndexBegin(self):
        return self.__begin

    @property
    def CharIndexEnd(self):
        return self.__end

    @property
    def Type(self):
        return self.__e_type
