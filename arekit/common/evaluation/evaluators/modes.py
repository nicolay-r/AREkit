from enum import Enum


class EvaluationModes(Enum):

    # Considering to evaluate opinions that were found by method
    Classification = 0

    # Considering to evaluate opinions that were found by method
    # and other opinions that was not found in result labeling.
    Extraction = 1
