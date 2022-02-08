class NerObjectDescriptor:

    def __init__(self, pos, length, obj_type):
        self.__pos = pos
        self.__len = length
        self.__obj_type = obj_type

    @property
    def Position(self):
        return self.__pos

    @property
    def Length(self):
        return self.__len

    @property
    def ObjectType(self):
        return self.__obj_type

    def get_range(self):
        return self.__pos, self.__pos + self.__len
