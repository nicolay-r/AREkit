# -*- coding: utf-8 -*-
from arekit.common.languages.pos import PartOfSpeechType
from arekit.common.languages.ru.cases import RussianCases, RussianCasesService
from arekit.common.languages.ru.number import RussianNumberType, RussianNumberTypeService
from arekit.common.languages.ru.pos_service import PartOfSpeechTypesService
from arekit.processing.pos.russian import RussianPOSTagger
from pymystem3 import Mystem


class POSMystemWrapper(RussianPOSTagger):

    _ArgsSeparator = ','
    _GrammarKey = 'gr'

    def __init__(self, mystem):
        assert(isinstance(mystem, Mystem))
        self.__mystem = mystem

    # region properties

    @property
    def POSCount(self):
        return PartOfSpeechTypesService.get_mystem_pos_count()

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
            return func(None)

        info = analysis['analysis']
        if len(info) == 0:
            return func(None)

        return func(info[0])

    @staticmethod
    def __get_pos(arguments):
        if arguments is None:
            return PartOfSpeechType.Unknown

        pos = arguments[POSMystemWrapper._GrammarKey].split(POSMystemWrapper._ArgsSeparator)[0]
        if '=' in pos:
            pos = pos.split('=')[0]

        return PartOfSpeechTypesService.get_mystem_from_string(pos)

    @staticmethod
    def __get_russian_case(arguments):
        if arguments is None:
            return RussianCases.UNKN

        all_params = set(POSMystemWrapper.__iter_params(arguments))

        for key, case in RussianCasesService.iter_rus_mystem_tags():
            if key in all_params:
                return case

        return RussianCases.UNKN

    @staticmethod
    def __get_number(arguments):
        if arguments is None:
            return RussianNumberType.UNKN

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

    def get_term_pos(self, term):
        assert(isinstance(term, unicode))
        analyzed = self.__mystem.analyze(term)
        return self.__extract_from_analysis(analyzed[0], self.__get_pos) \
            if len(analyzed) > 0 else PartOfSpeechType.Unknown

    def get_term_case(self, term):
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
        assert(isinstance(pos, PartOfSpeechType))
        return int(pos)

    def get_pos_as_unknown(self):
        return PartOfSpeechType.Unknown

    def is_adjective(self, pos_type):
        assert(isinstance(pos_type, PartOfSpeechType))
        return pos_type.upper() == PartOfSpeechType.ADJ

    def is_noun(self, pos_type):
        assert(isinstance(pos_type, PartOfSpeechType))
        return pos_type.upper() == PartOfSpeechType.NOUN

    def is_verb(self, pos_type):
        assert(isinstance(pos_type, PartOfSpeechType))
        return pos_type.upper() == PartOfSpeechType.VERB
