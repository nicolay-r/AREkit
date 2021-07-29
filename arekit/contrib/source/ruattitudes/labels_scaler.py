from arekit.contrib.source.ruattitudes.conts import \
    NEG_INT_VALUE, POS_INT_VALUE, \
    RU_ATTITUDES_POS_LABEL, RU_ATTITUDES_NEG_LABEL, NEU_INT_VALUE, RU_ATTITUDES_NEU_LABEL


class RuAttitudesLabelConverter:

    def __init__(self):

        self.__int_to_label_dict = {
                NEU_INT_VALUE: self._neutral_label_instance(),
                POS_INT_VALUE: self._positive_label_instance(),
                NEG_INT_VALUE: self._negative_label_instance()
        }

    @classmethod
    def _neutral_label_instance(cls):
        return RU_ATTITUDES_NEU_LABEL

    @classmethod
    def _positive_label_instance(cls):
        return RU_ATTITUDES_POS_LABEL

    @classmethod
    def _negative_label_instance(cls):
        return RU_ATTITUDES_NEG_LABEL

    def int_to_label(self, int_value):
        return self.__int_to_label_dict[int_value]
