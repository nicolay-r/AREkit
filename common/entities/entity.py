class Entity(object):

    def __init__(self, value):
        assert(type(value) == unicode and len(value) > 0)
        self.__value = value.lower()

    @property
    def Value(self):
        return self.__value
