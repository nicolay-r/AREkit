from arekit.common.entities.base import Entity
from arekit.common.entities.types import OpinionEntityType
from arekit.contrib.utils.entities.filter import EntityFilter


class EntityHelper(object):
    """ Named Entities formatting in text.
        Based on OntoNotes5 collection tags:
        https://catalog.ldc.upenn.edu/LDC2013T19
    """

    AGE = "AGE"
    AWARD = "AWARD"
    CITY = "CITY"
    COUNTRY = "COUNTRY"
    CRIME = "CRIME"
    DATE = "DATE"
    DISEASE = "DISEASE"
    DISTRICT = "DISTRICT"
    EVENT = "EVENT"
    FACILITY = "FACILITY"
    FAMILY = "FAMILY"
    IDEOLOGY = "IDEOLOGY"
    LANGUAGE = "LANGUAGE"
    LAW = "LAW"
    LOCATION = "LOCATION"
    MONEY = "MONEY"
    NATIONALITY = "NATIONALITY"
    NUMBER = "NUMBER"
    ORDINAL = "ORDINAL"
    ORGANIZATION = "ORGANIZATION"
    PENALTY = "PENALTY"
    PERCENT = "PERCENT"
    PERSON = "PERSON"
    PRODUCT = "PRODUCT"
    PROFESSION = "PROFESSION"
    RELIGION = "RELIGION"
    STATE_OR_PROVINCE = "STATE_OR_PROVINCE"
    TIME = "TIME"
    WORK_OF_ART = "WORK_OF_ART"


class SentiNerelEntityFilter(EntityFilter):
    """ Filter, oriented on sentiment related extraction task
        within SentiNEREL dataset.
    """

    def is_ignored(self, entity, e_type):
        """ Subject and Object could be one of the following object types:
                [PERSON, ORGANIZATION, COUNTRY, PROFESSION]
        """
        assert(isinstance(entity, Entity))
        assert(isinstance(e_type, OpinionEntityType))

        supported = [EntityHelper.PERSON, EntityHelper.ORGANIZATION, EntityHelper.COUNTRY, EntityHelper.PROFESSION]

        if e_type == OpinionEntityType.Subject:
            return entity.Type not in supported
        if e_type == OpinionEntityType.Object:
            return entity.Type not in supported
        else:
            return True