from collections import OrderedDict
from arekit.common.labels.base import Label
from arekit.common.labels.scaler import BaseLabelScaler
from arekit.contrib.source.ruattitudes.conts import \
    NEG_INT_VALUE, POS_INT_VALUE, \
    RU_ATTITUDES_POS_LABEL, RU_ATTITUDES_NEG_LABEL, NEU_INT_VALUE, RU_ATTITUDES_NEU_LABEL


class RuAttitudesLabelScaler(BaseLabelScaler):

    def __init__(self):

        neu_label = self._neutral_label_instance()
        pos_label = self._positive_label_instance()
        neg_label = self._negative_label_instance()

        assert(neu_label, Label)
        assert(pos_label, Label)
        assert(neg_label, Label)

        int_labels = [(neu_label, NEU_INT_VALUE),
                      (pos_label, POS_INT_VALUE),
                      (neg_label, NEG_INT_VALUE)]

        # TODO. Fix this.
        # TODO. This is useless. It was added as the base class expected non-null parameter.
        # TODO. Remove this in further updates.
        uint_labels = [(neu_label, 0),
                       (pos_label, 1),
                       (neg_label, 2)]

        super(RuAttitudesLabelScaler, self).__init__(uint_dict=OrderedDict(uint_labels),
                                                     int_dict=OrderedDict(int_labels))

    @classmethod
    def _neutral_label_instance(cls):
        return RU_ATTITUDES_NEU_LABEL

    @classmethod
    def _positive_label_instance(cls):
        return RU_ATTITUDES_POS_LABEL

    @classmethod
    def _negative_label_instance(cls):
        return RU_ATTITUDES_NEG_LABEL

