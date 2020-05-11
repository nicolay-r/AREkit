class Label:

    # TODO. Remove.
    @staticmethod
    def from_str(value):
        for l in Label._get_supported_labels():
            if l.to_str() == value:
                return l
        raise Exception("Label by value '{}' doesn't supported".format(value))

    # TODO. Remove.
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

    def __eq__(self, other):
        assert(isinstance(other, Label))
        return self.to_str() == other.to_str()

    def __ne__(self, other):
        assert(isinstance(other, Label))
        return self.to_str() != other.to_str()

    def __hash__(self):
        return hash(self.to_str())


class PositiveLabel(Label):

    def to_str(self):
        return 'pos'


class NegativeLabel(Label):

    def to_str(self):
        return 'neg'


class NeutralLabel(Label):

    def to_str(self):
        return 'neu'
