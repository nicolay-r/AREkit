from arekit.common.frames.variants.base import FrameVariant


class TextFrameVariant(object):
    """
    FrameVariant in a text, i.e. related object provides position in text.
    """

    def __init__(self, variant):
        assert(isinstance(variant, FrameVariant))
        self.__variant = variant
        self.__is_negated = False

    # region properties

    @property
    def Variant(self):
        return self.__variant

    @property
    def IsNegated(self):
        return self.__is_negated

    # endregion

    # region public methods

    def iter_terms(self):
        for term in self.__variant.iter_terms():
            yield term

    def set_is_negated(self, value):
        assert(isinstance(value, bool))
        self.__is_negated = value

    # endregion

    # region overriden

    def __len__(self):
        return len(self.__variant)

    # endregion