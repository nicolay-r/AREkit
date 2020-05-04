from collections import OrderedDict

from arekit.common.labels.base import Label


class BaseLabelScaler(object):

    def __init__(self, to_uint):
        assert(isinstance(to_uint, OrderedDict))
        self.__to_uint = to_uint
        self.__ordered_labels = list(to_uint.iterkeys())

    def ordered_suppoted_labels(self):
        return self.__ordered_labels

    def label_to_uint(self, label):
        assert(isinstance(label, Label))
        return self.__to_uint[label]

    def uint_to_label(self, value):
        assert(isinstance(value, int))
        assert(value >= 0)
        for label, v in self.__to_uint.iteritems():
            if v == value:
                return label
