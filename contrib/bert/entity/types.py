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
    def __iter_supported_names():
        return iter(BertEntityFormattersService.__names.keys())

    @staticmethod
    def get_type_by_name(name):
        return BertEntityFormattersService.__names[name]

    @staticmethod
    def find_name_by_type(entity_fmt_type):
        assert(isinstance(entity_fmt_type, BertEntityFormatterTypes))

        for name in BertEntityFormattersService.__iter_supported_names():
            related_type = BertEntityFormattersService.__names[name]
            if related_type == entity_fmt_type:
                return name

    @staticmethod
    def iter_supported_names():
        return BertEntityFormattersService.__iter_supported_names()
