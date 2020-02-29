class LabelsHelper(object):

    @staticmethod
    def create_label_from_uint(label_uint):
        raise NotImplementedError()

    @staticmethod
    def create_label_from_text_opinions(text_opinion_labels, label_creation_mode):
        raise NotImplementedError()

    @staticmethod
    def create_label_from_opinions(forward, backward):
        raise NotImplementedError()

    @staticmethod
    def create_opinions_from_text_opinion_and_label(text_opinion, label):
        raise NotImplementedError()

    @staticmethod
    def get_classes_count():
        raise NotImplementedError()


class LabelCalculationMode:
    FIRST_APPEARED = u'take_first_appeared'
    AVERAGE = u'average'