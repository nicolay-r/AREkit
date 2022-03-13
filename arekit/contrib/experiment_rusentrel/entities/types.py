from enum import Enum

from arekit.common.utils import EnumConversionService


class EntityFormatterTypes(Enum):

    RussianSimple = 1
    RussianCased = 2
    Simple = 3
    SimpleUppercase = 4
    SimpleSharpPrefixed = 5


class EntityFormattersService(EnumConversionService):

    _data = {
        "rus-cased-fmt": EntityFormatterTypes.RussianCased,
        'rus-simple': EntityFormatterTypes.RussianSimple,
        'simple-uppercase': EntityFormatterTypes.SimpleUppercase,
        'simple': EntityFormatterTypes.Simple,
        'sharp-simple': EntityFormatterTypes.SimpleSharpPrefixed
    }
