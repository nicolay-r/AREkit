from arekit.common.frames.variants.base import FrameVariant


class TextFrameVariant(object):
    """
    FrameVariant in a text, i.e. related object provides position in text.
    """

    def __init__(self, variant, is_inverted):
        assert(isinstance(variant, FrameVariant))
        assert(isinstance(is_inverted, bool))
        self.__variant = variant
        self.__is_inverted = is_inverted

    # region properties

    @property
    def Variant(self):
        return self.__variant

    @property
    def IsInverted(self):
        return self.__is_inverted

    # endregion

    # region public methods

    def iter_terms(self):
        for term in self.__variant.iter_terms():
            yield term

    # endregion

    # region overriden

    def __len__(self):
        return len(self.__variant)

    # endregion