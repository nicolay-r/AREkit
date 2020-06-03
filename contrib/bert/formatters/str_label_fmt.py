# -*- coding: utf-8 -*-
from arekit.common.labels.base import NegativeLabel, NeutralLabel, PositiveLabel
from arekit.common.labels.str_fmt import StringLabelsFormatter


class RussianThreeScaleLabelsFormatter(StringLabelsFormatter):

    def __init__(self):

        stol = {u'негативно': NegativeLabel(),
                u'позитивно': PositiveLabel(),
                u'нейтрально': NeutralLabel()}

        super(RussianThreeScaleLabelsFormatter, self).__init__(stol=stol)
