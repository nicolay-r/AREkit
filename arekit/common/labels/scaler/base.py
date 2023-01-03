from collections import OrderedDict

from arekit.common.labels.base import NoLabel, Label


class BaseLabelScaler(object):
    """ NOTE:
        Scaler -- set up conversion from int/uint to label and vice versa.
    """

    def __init__(self, uint_dict, int_dict):
        assert(isinstance(uint_dict, OrderedDict))
        assert(isinstance(int_dict, OrderedDict))
        assert(len(uint_dict) == len(int_dict))

        self.__uint_dict = uint_dict
        self.__int_dict = int_dict

        self.__ordered_labels = list(uint_dict.keys())
        self.__no_label_instance = self.__find_no_label_instance(iter(uint_dict.keys()))

    @property
    def LabelsCount(self):
        return len(self.__uint_dict)

    def ordered_suppoted_labels(self):
        return self.__ordered_labels

    def get_no_label_instance(self):
        if self.__no_label_instance is None:
            raise Exception("NoLabel does no supported by this scaler")

        return self.__no_label_instance

    # region private methods

    @staticmethod
    def __find_no_label_instance(labels_it):
        for label in labels_it:
            if isinstance(label, NoLabel):
                return label
        return None

    @staticmethod
    def __ltoi(label, d):
        assert(isinstance(label, Label))
        assert(isinstance(d, OrderedDict))
        return d[label]

    @staticmethod
    def __itol(value, d):
        assert(isinstance(value, int))
        assert(isinstance(d, OrderedDict))
        for label, v in d.items():
            if v == value:
                return label

    @staticmethod
    def __has_value(value, d):
        assert(isinstance(value, int))
        assert(isinstance(d, OrderedDict))
        for label, v in d.items():
            if v == value:
                return True
        return False

    # endregion

    def classes_count(self):
        return len(self.__uint_dict)

    def label_to_uint(self, label):
        return self.__ltoi(label, self.__uint_dict)

    def label_to_int(self, label):
        return self.__ltoi(label, self.__int_dict)

    def uint_to_label(self, value):
        return self.__itol(value, self.__uint_dict)

    def int_to_label(self, value):
        return self.__itol(value, self.__int_dict)

    def support_int_value(self, value):
        return self.__has_value(value, self.__int_dict)