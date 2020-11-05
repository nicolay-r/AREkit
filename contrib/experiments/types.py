from enum import Enum


class ExperimentTypes(Enum):

    RuSentRel = 1
    RuAttitudes = 2
    RuSentRelWithRuAttitudes = 3


class ExperimentTypesService:

    __names = {
        u'rsr': ExperimentTypes.RuSentRel,
        u'ra': ExperimentTypes.RuAttitudes,
        u'rsr+ra': ExperimentTypes.RuSentRelWithRuAttitudes
    }

    @staticmethod
    def get_type_by_name(name):
        return ExperimentTypesService.__names[name]

    @staticmethod
    def iter_supported_names():
        return iter(ExperimentTypesService.__names.keys())

