# -*- coding: utf-8 -*-
from arekit.common.languages.ru.cases import RussianCases, RussianCasesService
from arekit.common.languages.ru.number import RussianNumberType, RussianNumberTypeService
from arekit.processing.pos.russian import RussianPOSTagger
from arekit.processing.pos.base import POSTagger
from pymystem3 import Mystem


class POSMystemWrapper(RussianPOSTagger):

    PosAdjective = u"A"
    PosNoun = u"S"
    PosVerb = u"V"
    _ArgsSeparator = ','
    _GrammarKey = 'gr'

    # TODO. POS names implement as enum and move into the language folder (root, not russian)
    __pos_names = [PosNoun,
                   u"ADV",
                   u"ADVPRO",
                   u"ANUM",
                   u"APRO",
                   u"COM",
                   u"CONJ",
                   u"INTJ",
                   u"NUM",
                   u"PART",
                   u"PR",
                   PosAdjective,
                   u"SPRO",
                   PosVerb,
                   POSTagger.Unknown,
                   POSTagger.Empty]

    def __init__(self, mystem):
        assert(isinstance(mystem, Mystem))
        self.__mystem = mystem

    # region properties

    @property
    def POSCount(self):
        return len(self.__pos_names)

    # endregion

    # region private methods

    def __extract_from_analysis(self, analysis, func):
        """
        part of speech description:
            https://tech.yandex.ru/mystem/doc/grammemes-values-docpage/
        func: f(args) -> out
        returns: str or None
        """
        assert(callable(func))

        if 'analysis' not in analysis:
            return self.Unknown

        info = analysis['analysis']
        if len(info) == 0:
            return self.Unknown

        return func(info[0])

    @staticmethod
    def __get_pos(arguments):
        pos = arguments[POSMystemWrapper._GrammarKey].split(POSMystemWrapper._ArgsSeparator)[0]
        if '=' in pos:
            pos = pos.split('=')[0]
        return pos

    @staticmethod
    def __get_russian_case(arguments):
        all_params = set(POSMystemWrapper.__iter_params(arguments))

        for key, case in RussianCasesService.iter_rus_mystem_tags():
            if key in all_params:
                return case

        return RussianCases.UNKN

    @staticmethod
    def __get_number(arguments):
        all_params = set(POSMystemWrapper.__iter_params(arguments))

        for key, case in RussianNumberTypeService.iter_rus_mystem_tags():
            if key in all_params:
                return case

        return RussianNumberType.UNKN

    @staticmethod
    def __iter_params(arguments):
        params = arguments[POSMystemWrapper._GrammarKey].split(POSMystemWrapper._ArgsSeparator)
        for optionally_combined in params:
            for param in optionally_combined.split('='):
                yield param

    # endregion

    def get_terms_pos(self, terms):
        """ list of part of speech according to the certain word in text
        """
        assert(isinstance(terms, list))
        pos_list = []
        for term in terms:
            analyzed = self.__mystem.analyze(term)
            pos = self.__extract_from_analysis(analyzed[0], self.__get_pos) if len(analyzed) > 0 else self.Unknown
            pos_list.append(pos)

        return pos_list

    def get_term_pos(self, term):
        assert(isinstance(term, unicode))
        analyzed = self.__mystem.analyze(term)
        return self.__extract_from_analysis(analyzed[0], self.__get_pos) \
            if len(analyzed) > 0 else self.Unknown

    def get_terms_russian_cases(self, text):
        """ list of part of speech according to the certain word in text
        """
        assert(isinstance(text, unicode))
        cases = []

        analyzed = self.__mystem.analyze(text)
        for a in analyzed:
            pos = self.__extract_from_analysis(a, self.__get_russian_case) if len(analyzed) > 0 else RussianCases.Unknown
            cases.append(pos)

        return cases

    def get_term_russian_case(self, term):
        assert(isinstance(term, unicode))
        analyzed = self.__mystem.analyze(term)
        return self.__extract_from_analysis(analyzed[0], self.__get_russian_case) \
            if len(analyzed) > 0 else RussianCases.UNKN

    def get_term_number(self, term):
        assert(isinstance(term, unicode))
        analyzed = self.__mystem.analyze(term)
        return self.__extract_from_analysis(analyzed[0], self.__get_number) \
            if len(analyzed) > 0 else RussianNumberType.UNKN

    def pos_to_int(self, pos):
        assert(isinstance(pos, unicode))
        pos = pos.upper()
        return self.__pos_names.index(pos)

    def is_adjective(self, pos_type):
        assert(isinstance(pos_type, unicode))
        return pos_type.upper() == self.PosAdjective

    def is_noun(self, pos_type):
        assert(isinstance(pos_type, unicode))
        return pos_type.upper() == self.PosNoun

    def is_verb(self, pos_type):
        assert(isinstance(pos_type, unicode))
        return pos_type.upper() == self.PosVerb
