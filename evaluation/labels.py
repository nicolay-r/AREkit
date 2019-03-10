

class Label:

    @staticmethod
    def from_str(value):
        for l in Label._get_supported_labels():
            if l.to_str() == value:
                return l
        raise Exception("Label by value '{}' doesn't supported".format(value))

    @staticmethod
    def from_int(value):
        assert(isinstance(value, int))
        for l in Label._get_supported_labels():
            if l.to_int() == value:
                return l
        raise Exception("Label by value '{}' doesn't supported".format(value))

    @staticmethod
    def from_uint(value):
        assert(isinstance(value, int) and value >= 0)
        for l in Label._get_supported_labels():
            if l.to_uint() == value:
                return l
        raise Exception("Label by unsigned value '{}' doesn't supported".format(value))

    @staticmethod
    def _get_supported_labels():
        supported_labels = [
            PositiveLabel(),
            NegativeLabel(),
            NeutralLabel()
        ]
        return supported_labels

    def to_str(self):
        raise Exception("Not implemented exception")

    def to_int(self):
        raise Exception("Not implemented exception")

    def to_uint(self):
        raise Exception("Not implemented exception")

    def __eq__(self, other):
        assert(isinstance(other, Label))
        return self.to_int() == other.to_int()

    def __ne__(self, other):
        assert(isinstance(other, Label))
        return self.to_int() != other.to_int()


class PositiveLabel(Label):

    def to_str(self):
        return 'pos'

    def to_int(self):
        return int(1)

    def to_uint(self):
        return int(1)


class NegativeLabel(Label):

    def to_str(self):
        return 'neg'

    def to_int(self):
        return int(-1)

    def to_uint(self):
        return int(2)


class NeutralLabel(Label):

    def to_str(self):
        return 'neu'

    def to_int(self):
        return int(0)

    def to_uint(self):
        return int(0)


class LabelPair(Label):

    def __init__(self, forward, backward):
        assert(isinstance(forward, Label))
        assert(isinstance(backward, Label))
        self._forward = forward
        self._backward = backward

    @property
    def Forward(self):
        return self._forward

    @property
    def Backward(self):
        return self._backward

    @staticmethod
    def _pair_to_int(i, j):
        return int("{}{}".format(i, j), 3)

    def to_uint(self):
        return self._pair_to_int(self._forward.to_uint(), self._backward.to_uint())

    def to_int(self):
        return self.to_uint()

    def from_uint(self, value):
        for i in range(3):
            for j in range(3):
                if value == self._pair_to_int(i, j):
                    self._forward.from_int(i)
                    self._backward.from_int(j)

    def from_int(self, value):
        return self.from_uint(value)

