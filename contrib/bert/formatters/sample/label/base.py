class LabelProvider(object):

    def __init__(self, supported_labels):
        assert(isinstance(supported_labels, list))
        self.__labels = supported_labels

    @property
    def SupportedLabels(self):
        return self.__labels

    @staticmethod
    def get_label(expected_label, etalon_label):
        raise NotImplementedError()

    def get_supported_labels(self):
        raise NotImplementedError()
