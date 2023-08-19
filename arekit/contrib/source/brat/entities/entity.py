from arekit.common.entities.base import Entity


class BratEntity(Entity):
    """ Annotated entity in Brat-based collection corpus.
        Provides bounds, i.e. char indices in related sentence.
    """

    def __init__(self, id_in_doc, e_type, index_begin, index_end, value, childs, display_value=None, group_index=None):
        """ index_begin: int
                - char index (in case of string type of `text`)
                - term index (in case of list type of `text`)
            index_end: int
                - char index (in case of string type of `text`)
                - term index (in case of list type of `text`)
        """
        assert(isinstance(e_type, str))
        assert(isinstance(index_begin, int))
        assert(isinstance(index_end, int))
        super(BratEntity, self).__init__(value=value, e_type=e_type, childs=childs,
                                         display_value=display_value, group_index=group_index)

        self.__e_type = e_type
        self.__begin = index_begin
        self.__end = index_end
        self.__id = id_in_doc

    @property
    def IndexBegin(self):
        return self.__begin

    @property
    def IndexEnd(self):
        return self.__end

    @property
    def Type(self):
        return self.__e_type

    @property
    def ID(self):
        return self.__id
