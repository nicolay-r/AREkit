from collections import OrderedDict

from arekit.common.labels.scaler.base import BaseLabelScaler
from arekit.contrib.source.ruattitudes.conts import \
    RU_ATTITUDES_POS_LABEL, RU_ATTITUDES_NEG_LABEL, RU_ATTITUDES_NEU_LABEL


class RuAttitudesLabelScaler(BaseLabelScaler):

    def __init__(self):

        self.__int_to_label_dict = OrderedDict([
            (self._neutral_label_instance(), 0),
            (self._positive_label_instance(), 1),
            (self._negative_label_instance(), -1)])

        self.__uint_to_label_dict = OrderedDict([
            (self._neutral_label_instance(), 0),
            (self._positive_label_instance(), 1),
            (self._negative_label_instance(), 2)])

        super(RuAttitudesLabelScaler, self).__init__(int_dict=self.__int_to_label_dict,
                                                     uint_dict=self.__uint_to_label_dict)

    @classmethod
    def _neutral_label_instance(cls):
        return RU_ATTITUDES_NEU_LABEL

    @classmethod
    def _positive_label_instance(cls):
        return RU_ATTITUDES_POS_LABEL

    @classmethod
    def _negative_label_instance(cls):
        return RU_ATTITUDES_NEG_LABEL
