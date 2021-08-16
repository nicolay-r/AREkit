from enum import Enum


class RussianCases(Enum):
    """ Падежи русского языка
    """

    """ не определено
    """
    UNKN = 10

    """ именительный
    """
    NOM = 1

    """ родительный
    """
    GEN = 2

    """ дательный
    """
    DAT = 3

    """ винительный
    """
    ACC = 4

    """ творительный
    """
    INS = 5

    """ предложный
    """
    ABL = 6

    """ партитив
    """
    PART = 7

    """ местный 
    """
    LOC = 8

    """ звательный
    """
    VOC = 9


class RussianCasesService(object):

    __english = {
        'nom': RussianCases.NOM,
        'gen': RussianCases.GEN,
        'dat': RussianCases.DAT,
        'acc': RussianCases.ACC,
        'ins': RussianCases.INS,
        'abl': RussianCases.ABL,
        'part': RussianCases.PART,
        'loc': RussianCases.LOC,
        'voc': RussianCases.VOC,
    }

    __mystem_russian = {
        'им': RussianCases.NOM,
        'род': RussianCases.GEN,
        'дат': RussianCases.DAT,
        'вин': RussianCases.ACC,
        'твор': RussianCases.INS,
        'пр': RussianCases.ABL,
        'парт': RussianCases.PART,
        'местн': RussianCases.LOC,
        'зват': RussianCases.VOC,
    }

    @staticmethod
    def iter_rus_mystem_tags():
        for key, value in RussianCasesService.__mystem_russian.items():
            yield key, value
