from enum import Enum


class FoldingType(Enum):
    """
    Assumes a fixed separation onto train and test collections
    """
    Fixed = 1

    """
    Assumes separation using k-fold cross-validation approach
    """
    CrossValidation = 2

    @staticmethod
    def from_str(value):
        for t in FoldingType:
            if t.value == value:
                return t
