from arekit.common.labels.base import Label


class StringLabelsFormatter(object):

    def __init__(self, stol):
        assert(isinstance(stol, dict))
        self._stol = stol

    def str_to_label(self, value):
        assert(isinstance(value, unicode))
        assert(value in self._stol)
        return self._stol[value]

    def label_to_str(self, label):
        assert(isinstance(label, Label))
        for value, l in self._stol.iteritems():
            if l == label:
                return value

