from enum import Enum


class RussianNumberType(Enum):

    UNKN = 3

    Plural = 1

    Single = 2


class RussianNumberTypeService(object):

    __russian = {
        'ед': RussianNumberType.Single,
        'мн': RussianNumberType.Plural
    }

    @staticmethod
    def iter_rus_mystem_tags():
        for key, value in RussianNumberTypeService.__russian.items():
            yield key, value
