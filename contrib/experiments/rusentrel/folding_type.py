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
