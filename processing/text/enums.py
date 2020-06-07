from enum import Enum


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