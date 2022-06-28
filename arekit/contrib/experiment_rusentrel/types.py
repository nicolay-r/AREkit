from enum import Enum

from arekit.contrib.experiment_rusentrel.utils import EnumConversionService


class ExperimentTypes(Enum):

    RuSentRel = 1
    RuAttitudes = 2


class ExperimentTypesService(EnumConversionService):

    _data = {
        'rsr': ExperimentTypes.RuSentRel,
        'ra': ExperimentTypes.RuAttitudes
    }
