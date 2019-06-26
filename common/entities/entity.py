class Entity(object):

    def __init__(self, value, id_in_doc):
        assert(type(value) == unicode and len(value) > 0)
        self.__value = value.lower()
        self.__id = id_in_doc

    @property
    def Value(self):
        return self.__value

    @property
    def IdInDocument(self):
        return self.__id
