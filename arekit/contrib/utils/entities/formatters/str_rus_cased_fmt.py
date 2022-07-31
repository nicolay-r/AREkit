from arekit.common.entities.base import Entity
from arekit.common.entities.str_fmt import StringEntitiesFormatter
from arekit.common.entities.types import OpinionEntityType
from arekit.contrib.utils.processing.languages.ru.cases import RussianCases
from arekit.contrib.utils.processing.languages.ru.number import RussianNumberType
from arekit.contrib.utils.processing.pos.russian import RussianPOSTagger


class RussianEntitiesCasedFormatter(StringEntitiesFormatter):

    # Объект/Субъект
    obj_subj_cases_map = {
        RussianCases.UNKN: ['', ''],      # UNKN
        RussianCases.NOM: ['', "ы"],      # именительный
        RussianCases.GEN: ['а', 'ов'],    # родительный
        RussianCases.DAT: ['y', 'ам'],    # дательный
        RussianCases.ACC: ['', 'ы'],      # винительный
        RussianCases.INS: ['ом', 'aми'],  # творительный
        RussianCases.ABL: ['e', 'ах']     # предложный
    }

    # Сущност
    entity_cases_map = {
        RussianCases.UNKN: ['ь', 'и'],     # UNKN
        RussianCases.NOM: ['ь', "и"],      # именительный
        RussianCases.GEN: ['и', 'ей'],     # родительный
        RussianCases.DAT: ['и', 'ям'],     # дательный
        RussianCases.ACC: ['ь', 'и'],      # винительный
        RussianCases.INS: ['ью', 'ьями'],  # творительный
        RussianCases.ABL: ['и', 'ях']      # предложный
    }

    def __init__(self, pos_tagger):
        assert(isinstance(pos_tagger, RussianPOSTagger))
        self.__pos_tagger = pos_tagger

    def to_string(self, original_value, entity_type):
        assert(isinstance(original_value, Entity))
        assert(isinstance(entity_type, OpinionEntityType))

        template = None
        cases_map = None

        if (entity_type == OpinionEntityType.Object) or (entity_type == OpinionEntityType.SynonymObject):
            template = "объект"
            cases_map = self.obj_subj_cases_map
        elif (entity_type == OpinionEntityType.Subject) or (entity_type == OpinionEntityType.SynonymSubject):
            template = "субъект"
            cases_map = self.obj_subj_cases_map
        elif entity_type == OpinionEntityType.Other:
            template = "сущност"
            cases_map = self.entity_cases_map

        return self.__get_correct_declention(value=original_value.Value,
                                             template=template,
                                             cases_map=cases_map)

    def __get_correct_declention(self, value, template, cases_map):
        assert(isinstance(value, str))
        assert(isinstance(template, str))
        assert(isinstance(cases_map, dict))

        num = self.__pos_tagger.get_term_number(value)
        case = self.__pos_tagger.get_term_case(value)

        assert(isinstance(num, RussianNumberType))
        assert(isinstance(case, RussianCases))

        if num == RussianNumberType.UNKN or num == RussianNumberType.Single:
            num_int = 0
        else:
            num_int = 1

        if case not in cases_map:
            case = RussianCases.UNKN

        return template + (cases_map[case])[num_int]

