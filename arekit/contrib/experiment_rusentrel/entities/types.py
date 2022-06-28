from enum import Enum

from arekit.contrib.experiment_rusentrel.utils import EnumConversionService


class EntityFormatterTypes(Enum):

    RussianCased = 1
    HiddenSimpleRus = 2
    HiddenSimpleEng = 3
    HiddenSimpleUppercase = 4
    HiddenBertStyled = 5


class EntityFormattersService(EnumConversionService):

    _data = {
        "rus-cased-fmt": EntityFormatterTypes.RussianCased,
        'hidden-simple-eng': EntityFormatterTypes.HiddenSimpleEng,
        'hidden-simple-rus': EntityFormatterTypes.HiddenSimpleRus,
        'hidden-simple-rus-uppercase': EntityFormatterTypes.HiddenSimpleUppercase,
        'hidden-bert-styled': EntityFormatterTypes.HiddenBertStyled
    }
