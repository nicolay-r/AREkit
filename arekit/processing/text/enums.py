from enum import Enum


# TODO. To common.
class TermFormat(Enum):
    """
    Supported types of terms
    """

    """
    Original value
    """
    Raw = 1

    """
    Processed by stemmer
    """
    Lemma = 2