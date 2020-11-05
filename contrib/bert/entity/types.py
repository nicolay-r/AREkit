from enum import Enum


class BertEntityFormatterTypes(Enum):

    RussianSimple = 1
    RussianCased = 2
    Simple = 3


class BertEntityFormattersService:

    __names = {
        u"russian-cased-fmt": BertEntityFormatterTypes.RussianCased,
        u'russian-simple': BertEntityFormatterTypes.RussianSimple,
        u'simple': BertEntityFormatterTypes.Simple
    }

    @staticmethod
    def get_type_by_name(name):
        return BertEntityFormattersService.__names[name]

    @staticmethod
    def iter_supported_names():
        return iter(BertEntityFormattersService.__names.keys())
