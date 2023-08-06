import enum


class NerelVersions(enum.Enum):
    """ List of the supported version of this collection
    """

    V1 = "v1_0"
    V11 = "v1_1"


DEFAULT_VERSION = NerelVersions.V1
