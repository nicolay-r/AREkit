from enum import Enum


class FoldingType(Enum):
    """
    Assumes a fixed separation onto train and test collections
    """
    Fixed = u'fx'

    """
    Assumes separation using k-fold cross-validation approach
    """
    CrossValidation = u'cv'

    @staticmethod
    def from_str(value):
        for t in FoldingType:
            if t.value == value:
                return t
