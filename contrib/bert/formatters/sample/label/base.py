class LabelProvider(object):

    @staticmethod
    def get_label(expected_label, etalon_label):
        raise NotImplementedError()

    @staticmethod
    def get_supported_labels():
        raise NotImplementedError()
