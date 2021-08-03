# -*- coding: utf-8 -*-
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
        u'nom': RussianCases.NOM,
        u'gen': RussianCases.GEN,
        u'dat': RussianCases.DAT,
        u'acc': RussianCases.ACC,
        u'ins': RussianCases.INS,
        u'abl': RussianCases.ABL,
        u'part': RussianCases.PART,
        u'loc': RussianCases.LOC,
        u'voc': RussianCases.VOC,
    }

    __mystem_russian = {
        u'им': RussianCases.NOM,
        u'род': RussianCases.GEN,
        u'дат': RussianCases.DAT,
        u'вин': RussianCases.ACC,
        u'твор': RussianCases.INS,
        u'пр': RussianCases.ABL,
        u'парт': RussianCases.PART,
        u'местн': RussianCases.LOC,
        u'зват': RussianCases.VOC,
    }

    @staticmethod
    def iter_rus_mystem_tags():
        for key, value in RussianCasesService.__mystem_russian.iteritems():
            yield key, value
