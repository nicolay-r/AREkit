class Label(object):

    def __eq__(self, other):
        assert(isinstance(other, Label))
        return type(self) == type(other)

    def __ne__(self, other):
        assert(isinstance(other, Label))
        return type(self) != type(other)

    def __hash__(self):
        return hash(self.to_class_str())

    def to_class_str(self):
        return self.__class__.__name__


class NoLabel(Label):
    pass
