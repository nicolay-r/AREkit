from arekit.common.bound import Bound
from arekit.common.frame_variants.base import FrameVariant


class TextFrameVariant(object):
    """
    FrameVariant in a text, i.e. related object provides position in text.
    """

    def __init__(self, variant, start_index, is_inverted):
        assert(isinstance(variant, FrameVariant))
        assert(isinstance(start_index, int))
        assert(isinstance(is_inverted, bool))
        self.__variant = variant
        self.__start_index = start_index
        self.__is_inverted = is_inverted

    # region properties

    @property
    def Variant(self):
        return self.__variant

    @property
    def Position(self):
        return self.__start_index

    @property
    def IsInverted(self):
        return self.__is_inverted

    # endregion

    # region public methods

    def get_bound(self):
        return Bound(pos=self.__start_index, length=len(self))

    def iter_terms(self):
        for term in self.__variant.iter_terms():
            yield term

    # endregion

    # region overriden

    def __len__(self):
        return len(self.__variant)

    # endregion