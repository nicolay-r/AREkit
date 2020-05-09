class Label:

    @staticmethod
    def from_str(value):
        for l in Label._get_supported_labels():
            if l.to_str() == value:
                return l
        raise Exception("Label by value '{}' doesn't supported".format(value))

    # TODO. Move
    @staticmethod
    def from_int(value):
        assert(isinstance(value, int))
        for l in Label._get_supported_labels():
            if l.to_int() == value:
                return l
        raise Exception("Label by value '{}' doesn't supported".format(value))

    # TODO. Move
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
        raise NotImplementedError()

    # TODO. Move
    def to_int(self):
        raise NotImplementedError()

    # TODO. Move
    def to_uint(self):
        raise NotImplementedError()

    def __eq__(self, other):
        assert(isinstance(other, Label))
        # TODO. Use str instead
        return self.to_int() == other.to_int()

    def __ne__(self, other):
        assert(isinstance(other, Label))
        # TODO. Use str instead
        return self.to_int() != other.to_int()

    def __hash__(self):
        return hash(self.to_int())


class PositiveLabel(Label):

    def to_str(self):
        return 'pos'

    # TODO. Move to scaler
    def to_int(self):
        return int(1)

    # TODO. Move to scaler
    def to_uint(self):
        return int(1)


class NegativeLabel(Label):

    def to_str(self):
        return 'neg'

    # TODO. Move to scaler
    def to_int(self):
        return int(-1)

    # TODO. Move to scaler
    def to_uint(self):
        return int(2)


class NeutralLabel(Label):

    def to_str(self):
        return 'neu'

    # TODO. Move to scaler
    def to_int(self):
        return int(0)

    # TODO. Move to scaler
    def to_uint(self):
        return int(0)
