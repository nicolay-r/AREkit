from arekit.common.labels.base import Label


# TODO. This should be moded into formatter.py
class StringLabelsFormatter(object):
    """ NOTE:
        Set up convertion from string into label instance.
    """

    def __init__(self, stol):
        """ stol: string to label dictionary
                dictionary: string -> label_type
        """
        assert(isinstance(stol, dict))

        for key, value in stol.items():
            # Perfom parameters check.
            assert(isinstance(key, str))
            assert(issubclass(value, Label))

        self._stol = stol
        self.__supported_label_types = set(self._stol.values())

    def __is_label_type_supported(self, label):
        return label in self.__supported_label_types

    def str_to_label(self, value):
        assert(isinstance(value, str))

        if not value in self._stol:
            raise Exception("Label value `{}` is not supported.".format(value))

        label_type = self._stol[value]
        return label_type()

    def label_to_str(self, label):
        assert(isinstance(label, Label))

        label_type = type(label)

        if not self.__is_label_type_supported(label_type):
            raise Exception("Label type {label} is not supported. Supported labels: [{values}]".format(
                label=label_type, values=self.__supported_label_types))

        for value, supported_label_type in self._stol.items():
            if supported_label_type == label_type:
                return value

    def supports_label(self, label):
        return type(label) in self.__supported_label_types

    def supports_value(self, value):
        assert(isinstance(value, str))
        return value in self._stol

