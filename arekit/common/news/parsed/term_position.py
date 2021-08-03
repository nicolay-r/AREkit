from enum import Enum


class TermPosition:

    def __init__(self, term_ind_in_doc, term_ind_in_sent, s_ind):
        self.__t_ind_in_doc = term_ind_in_doc
        self.__t_ind_in_sent = term_ind_in_sent
        self.__s_ind = s_ind

    def get_index(self, position_type):
        assert(isinstance(position_type, TermPositionTypes))

        if position_type == TermPositionTypes.IndexInDocument:
            return self.__t_ind_in_doc
        if position_type == TermPositionTypes.IndexInSentence:
            return self.__t_ind_in_sent
        if position_type == TermPositionTypes.SentenceIndex:
            return self.__s_ind


class TermPositionTypes(Enum):

    """
    Corresponds to an index of a related term in a whole document
    (document considered as a sequence of terms)
    """
    IndexInDocument = 1

    """
    Corresponds to an index of a related term in a certain sentence.
    """
    IndexInSentence = 2

    """
    Corresponds to an index of a sentence in a whole document.
    """
    SentenceIndex = 3




