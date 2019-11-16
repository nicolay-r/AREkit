from core.common.labels.base import Label


class LabelPair(Label):

    def __init__(self, forward, backward):
        assert(isinstance(forward, Label))
        assert(isinstance(backward, Label))
        self.__forward = forward
        self.__backward = backward

    @property
    def Forward(self):
        return self.__forward

    @property
    def Backward(self):
        return self.__backward

    @classmethod
    def create_inverted(cls, other):
        assert(isinstance(other, LabelPair))
        return LabelPair(other.Backward, other.Forward)

    @staticmethod
    def _pair_to_int(i, j):
        return int("{}{}".format(i, j), 3)

    def to_uint(self):
        return self._pair_to_int(self.__forward.to_uint(), self.__backward.to_uint())

    def to_int(self):
        return self.to_uint()

    def to_str(self):
        return u"{}-{}".format(self.__forward.to_str(), self.__backward.to_str())

    @staticmethod
    def from_uint(value):
        for i in xrange(3):
            for j in xrange(3):
                if value == LabelPair._pair_to_int(i, j):
                    return LabelPair(Label.from_uint(i), Label.from_uint(j))

    @staticmethod
    def from_int(value):
        return LabelPair.from_uint(value)