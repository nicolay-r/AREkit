from arekit.common.labels.base import Label


# TODO. This should be moded into formatter.py
class StringLabelsFormatter(object):
    """ NOTE:
        Set up convertion from string into label instance.
    """

    def __init__(self, stol):
        assert(isinstance(stol, dict))
        self._stol = stol
        self.__supported_labels = set(self._stol.itervalues())

    def __is_label_supported(self, label):
        return label in self.__supported_labels

    def str_to_label(self, value):
        assert(isinstance(value, unicode))
        assert(value in self._stol)
        return self._stol[value]

    def label_to_str(self, label):
        assert(isinstance(label, Label))

        if not self.__is_label_supported(label):
            raise Exception("Label {label} is not supported. Supported labels: [{values}]".format(
                label=label, values=self.__supported_labels))

        for value, supported_label in self._stol.iteritems():
            if supported_label == label:
                return value

    def supports_label(self, label):
        return label in self.__supported_labels

    def supports_value(self, value):
        assert(isinstance(value, unicode))
        return value in self._stol

