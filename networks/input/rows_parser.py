import pandas as pd

from arekit.common.experiment import const
from arekit.common.experiment.scales.base import BaseLabelScaler


class ParsedSampleRow(object):
    """
    Provides a parsed information for a sample row.
    TODO. Use this class as API
    """

    def __init__(self, row, labels_scaler):
        assert(isinstance(row, pd.Series))
        assert(isinstance(labels_scaler, BaseLabelScaler))

        self.__sentiment = None

        for key, value in row.iteritems():
            if key == const.LABEL:
                self.__sentiment = labels_scaler.uint_to_label(value)
            if key == const.ID:
                self.__row_id = self.__to_unicode_term(value)
            if key == const.S_IND:
                self.__subj_ind = value
            if key == const.T_IND:
                self.__obj_ind = value
            if key == "text_a":
                self.__terms = [self.__to_unicode_term(term) for term in row["text_a"].split(' ')]

    @staticmethod
    def __to_unicode_term(term):
        return term if isinstance(term, unicode) else term.decode('utf-8')

    @property
    def SampleID(self):
        return self.__row_id
    
    @property
    def Terms(self):
        return self.__terms

    @property
    def SubjectIndex(self):
        return self.__subj_ind

    @property
    def ObjectIndex(self):
        return self.__obj_ind
    
    @property
    def Sentiment(self):
        return self.__sentiment

    @classmethod
    def parse(cls, row, labels_scaler):
        return cls(row=row,
                   labels_scaler=labels_scaler)
