from enum import Enum


class ExperimentTypes(Enum):

    RuSentRel = 1
    RuAttitudes = 2
    RuSentRelWithRuAttitudes = 3


class ExperimentTypesService:

    __names = {
        'rsr': ExperimentTypes.RuSentRel,
        'ra': ExperimentTypes.RuAttitudes,
        'rsr+ra': ExperimentTypes.RuSentRelWithRuAttitudes
    }

    @staticmethod
    def get_type_by_name(name):
        return ExperimentTypesService.__names[name]

    @staticmethod
    def iter_supported_names():
        return iter(list(ExperimentTypesService.__names.keys()))

