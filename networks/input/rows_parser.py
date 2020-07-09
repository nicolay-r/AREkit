# TODO: This should be generated from the samples formatter, neural network samples formatter.
# TODO: This should be generated from the samples formatter, neural network samples formatter.
# TODO: This should be generated from the samples formatter, neural network samples formatter.


class ParsedSampleRow(object):
    """
    Provides a parsed information for a sample row.
    TODO. Use this class as API
    """

    def __init__(self):
        self.__sentiment = None
        self.__row_id = None
        self.__subj_ind = None
        self.__obj_ind = None
        self.__terms = None

    @property
    def RowID(self):
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
    def parse(cls, row):
        assert(isinstance(row, list))
        return cls()
