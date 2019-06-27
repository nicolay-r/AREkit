class TermPosition:

    def __init__(self, doc_level_ind, s_level_ind, s_ind):
        self.__doc_level_ind = doc_level_ind
        self.__s_level_ind = s_level_ind
        self.__s_ind = s_ind

    @property
    def DocLevelIndex(self):
        return self.__doc_level_ind

    @property
    def SentenceLevelIndex(self):
        return self.__s_level_ind

    @property
    def SentenceIndex(self):
        return self.__s_ind

