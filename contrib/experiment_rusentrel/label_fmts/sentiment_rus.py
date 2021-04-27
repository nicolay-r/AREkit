# -*- coding: utf-8 -*-
from arekit.common.labels.base import NegativeLabel, NeutralLabel, PositiveLabel
from arekit.common.labels.str_fmt import StringLabelsFormatter


class RussianThreeScaleRussianLabelsFormatter(StringLabelsFormatter):
    """ NOTE:
        This class founds its application in language models, in NLI task.
        Therefore, this class is related to this experiment.
    """

    def __init__(self):

        stol = {u'негативно': NegativeLabel(),
                u'позитивно': PositiveLabel(),
                u'нейтрально': NeutralLabel()}

        super(RussianThreeScaleRussianLabelsFormatter, self).__init__(stol=stol)