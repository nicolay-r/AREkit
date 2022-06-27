from enum import Enum


class DataType(Enum):
    """
    Describes collection types that supportes in
    current implementation, and provides by collections.
    """

    Train = 1

    Test = 2

    Dev = 3

    Etalon = 4

