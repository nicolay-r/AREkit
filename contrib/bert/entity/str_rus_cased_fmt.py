# -*- coding: utf-8 -*-
from arekit.common.entities.base import Entity
from arekit.common.entities.str_mask_fmt import StringEntitiesFormatter
from arekit.common.entities.types import EntityType
from arekit.common.languages.ru.cases import RussianCases
from arekit.common.languages.ru.number import RussianNumberType
from arekit.processing.pos.russian import RussianPOSTagger


class RussianEntitiesCasedFormatter(StringEntitiesFormatter):

    # Объект/Субъект
    obj_subj_cases_map = {
        RussianCases.UNKN: [u'', u''],      # UNKN
        RussianCases.NOM: [u'', u"ы"],      # именительный
        RussianCases.GEN: [u'а', u'ов'],    # родительный
        RussianCases.DAT: [u'y', u'ам'],    # дательный
        RussianCases.ACC: [u'', u'ы'],      # винительный
        RussianCases.INS: [u'ом', u'aми'],  # творительный
        RussianCases.ABL: [u'e', u'ах']     # предложный
    }

    # Сущност
    entity_cases_map = {
        RussianCases.UNKN: [u'ь', u'и'],     # UNKN
        RussianCases.NOM: [u'ь', u"и"],      # именительный
        RussianCases.GEN: [u'и', u'ей'],     # родительный
        RussianCases.DAT: [u'и', u'ям'],     # дательный
        RussianCases.ACC: [u'ь', u'и'],      # винительный
        RussianCases.INS: [u'ью', u'ьями'],  # творительный
        RussianCases.ABL: [u'и', u'ях']      # предложный
    }

    def __init__(self, pos_tagger):
        assert(isinstance(pos_tagger, RussianPOSTagger))
        self.__pos_tagger = pos_tagger

    def to_string(self, original_value, entity_type):
        assert(isinstance(original_value, Entity))
        assert(isinstance(entity_type, EntityType))

        template = None
        cases_map = None

        if (entity_type == EntityType.Object) or (entity_type == EntityType.SynonymObject):
            template = u"объект"
            cases_map = self.obj_subj_cases_map
        elif (entity_type == EntityType.Subject) or (entity_type == EntityType.SynonymSubject):
            template = u"субъект"
            cases_map = self.obj_subj_cases_map
        elif entity_type == EntityType.Other:
            template = u"сущност"
            cases_map = self.entity_cases_map

        return self.__get_correct_declention(value=original_value.Value,
                                             template=template,
                                             cases_map=cases_map)

    def __get_correct_declention(self, value, template, cases_map):
        assert(isinstance(value, unicode))
        assert(isinstance(template, unicode))
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

