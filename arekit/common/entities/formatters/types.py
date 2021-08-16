from enum import Enum


class EntityFormatterTypes(Enum):

    RussianSimple = 1
    RussianCased = 2
    Simple = 3
    SimpleUppercase = 4
    SimpleSharpPrefixed = 5


class EntityFormattersService:

    __names = {
        "rus-cased-fmt": EntityFormatterTypes.RussianCased,
        'rus-simple': EntityFormatterTypes.RussianSimple,
        'simple-uppercase': EntityFormatterTypes.SimpleUppercase,
        'simple': EntityFormatterTypes.Simple,
        'sharp-simple': EntityFormatterTypes.SimpleSharpPrefixed,
    }

    @staticmethod
    def __iter_supported_names():
        return iter(list(EntityFormattersService.__names.keys()))

    @staticmethod
    def get_type_by_name(name):
        return EntityFormattersService.__names[name]

    @staticmethod
    def find_name_by_type(entity_fmt_type):
        assert(isinstance(entity_fmt_type, EntityFormatterTypes))

        for name in EntityFormattersService.__iter_supported_names():
            related_type = EntityFormattersService.__names[name]
            if related_type == entity_fmt_type:
                return name

    @staticmethod
    def iter_supported_names():
        return EntityFormattersService.__iter_supported_names()
