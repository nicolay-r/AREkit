class Bound:

    def __init__(self, pos, length):
        assert(isinstance(pos, int))
        assert(isinstance(length, int))
        self.__pos = pos
        self.__length = length

    # region properties

    @property
    def Position(self):
        return self.__pos

    @property
    def Length(self):
        return self.__length

    # endregion

    def itersects_with(self, other):
        begin = self.__pos
        end = self.__pos + self.__length
        other_begin = other.Position
        other_end_included = other.Position + other.Length - 1
        if end > other_begin >= begin:
            return True
        if end > other_end_included >= begin:
            return True
        if other_begin < begin and end <= other_end_included:
            return True
        return False

    def intersect(self, other):
        begin = self.__pos
        end = self.__pos + self.__length
        other_begin = other.Position
        other_end = other.Position + other.Length
        actual_begin = min(begin, other_begin)
        actual_length = max(end, other_end) - actual_begin
        return Bound(pos=actual_begin, length=actual_length)

    def contains(self, other):
        begin = self.__pos
        end = self.__pos + self.__length
        other_begin = other.Position
        other_end = other.Position + other.Length
        return begin <= other_begin and end >= other_end
