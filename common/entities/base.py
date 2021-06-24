class Entity(object):

    def __init__(self, value, e_type, id_in_doc, group_index=None):
        assert(isinstance(value, unicode) and len(value) > 0)
        if not isinstance(e_type, unicode):
            print e_type
        assert(isinstance(e_type, unicode))
        assert(isinstance(group_index, int) or group_index is None)
        self.__value = value.lower()
        self.__id = id_in_doc
        self.__type = e_type
        self.__group_index = group_index

    @property
    def GroupIndex(self):
        return self.__group_index

    @property
    def Value(self):
        return self.__value

    @property
    def IdInDocument(self):
        return self.__id

    @property
    def Type(self):
        return self.__type
