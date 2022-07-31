from pymystem3 import Mystem

from arekit.contrib.utils.processing.languages.pos import PartOfSpeechType
from arekit.contrib.utils.processing.languages.ru.cases import RussianCases, RussianCasesService
from arekit.contrib.utils.processing.languages.ru.number import RussianNumberType, RussianNumberTypeService
from arekit.contrib.utils.processing.languages.ru.pos_service import PartOfSpeechTypesService
from arekit.contrib.utils.processing.pos.russian import RussianPOSTagger


class POSMystemWrapper(RussianPOSTagger):

    _ArgsSeparator = ','
    _GrammarKey = 'gr'

    def __init__(self, mystem):
        assert(isinstance(mystem, Mystem))
        self.__mystem = mystem

    # region private methods

    @staticmethod
    def __extract_from_analysis(analysis, func):
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
        assert(isinstance(term, str))
        analyzed = self.__mystem.analyze(term)
        return self.__extract_from_analysis(analyzed[0], self.__get_pos) \
            if len(analyzed) > 0 else PartOfSpeechType.Unknown

    def get_term_case(self, term):
        assert(isinstance(term, str))
        analyzed = self.__mystem.analyze(term)
        return self.__extract_from_analysis(analyzed[0], self.__get_russian_case) \
            if len(analyzed) > 0 else RussianCases.UNKN

    def get_term_number(self, term):
        assert(isinstance(term, str))
        analyzed = self.__mystem.analyze(term)
        return self.__extract_from_analysis(analyzed[0], self.__get_number) \
            if len(analyzed) > 0 else RussianNumberType.UNKN

    def get_terms_russian_cases(self, text):
        """ list of part of speech according to the certain word in text
        """
        assert(isinstance(text, str))
        cases = []

        analyzed = self.__mystem.analyze(text)
        for a in analyzed:
            pos = self.__extract_from_analysis(a, self.__get_russian_case) if len(analyzed) > 0 else RussianCases.UNKN
            cases.append(pos)

        return cases

    def pos_to_int(self, pos):
        assert(isinstance(pos, PartOfSpeechType))
        return int(pos)

    @staticmethod
    def is_adjective(pos_type):
        assert(isinstance(pos_type, PartOfSpeechType))
        return pos_type == PartOfSpeechType.ADJ

    @staticmethod
    def is_noun(pos_type):
        assert(isinstance(pos_type, PartOfSpeechType))
        return pos_type == PartOfSpeechType.NOUN

    @staticmethod
    def is_verb(pos_type):
        assert(isinstance(pos_type, PartOfSpeechType))
        return pos_type == PartOfSpeechType.VERB
