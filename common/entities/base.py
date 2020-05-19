class Entity(object):

    def __init__(self, value, e_type, id_in_doc):
        assert(isinstance(value, unicode) and len(value) > 0)
        assert(isinstance(e_type, unicode))
        self.__value = value.lower()
        self.__id = id_in_doc
        self.__type = e_type

    @property
    def Value(self):
        return self.__value

    @property
    def IdInDocument(self):
        return self.__id

    @property
    def Type(self):
        return self.__type
