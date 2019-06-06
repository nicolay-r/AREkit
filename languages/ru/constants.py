# -*- coding: utf-8 -*-
class RussianConstants:

    __prepositions = {u'к', u'на', u'по', u'с', u'до', u'в', u'во'}

    def __init__(self):
        pass

    @property
    def PrepositionSet(self):
        return self.__prepositions
