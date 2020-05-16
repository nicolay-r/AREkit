class StringLabelsFormatter(object):

    def __init__(self, stol):
        assert(isinstance(stol, dict))
        self.__stol = stol

    def str_to_label(self, value):
        assert(isinstance(value, unicode))
        return self.__stol[value]

    def label_to_str(self, value):
        assert(isinstance(value, int))
        for label, v in self.__stol.iteritems():
            if v == value:
                return label

