class Bound:

    def __init__(self, pos, length):
        assert(isinstance(pos, int))
        assert(isinstance(length, int))
        self.__pos = pos
        self.__length = length

    @property
    def Position(self):
        return self.__pos

    @property
    def Length(self):
        return self.__length
