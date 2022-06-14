class BratRelation(object):

    def __init__(self, id_in_doc, source_id, target_id, rel_type):
        assert(isinstance(id_in_doc, str))
        assert(isinstance(source_id, int))
        assert(isinstance(target_id, int))
        assert(isinstance(rel_type, str))

        self.__id = id_in_doc
        self.__rel_type = rel_type
        self.__source_id = source_id
        self.__target_id = target_id

    @property
    def ID(self):
        return self.__id

    @property
    def Type(self):
        return self.__rel_type

    @property
    def SourceID(self):
        """ Arg0.
        """
        return self.__source_id

    @property
    def TargetID(self):
        """ Arg1.
        """
        return self.__target_id
