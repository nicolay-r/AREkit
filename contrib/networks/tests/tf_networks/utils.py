import numpy as np

from arekit.common.languages.ru.pos_service import PartOfSpeechTypesService
from arekit.contrib.networks.context.configurations.base.base import DefaultNetworkConfig


def init_config(config):
    assert(isinstance(config, DefaultNetworkConfig))
    config.modify_classes_count(3)
    config.set_term_embedding(np.zeros((100, 100)))
    config.set_class_weights([1] * config.ClassesCount)
    config.set_pos_count(PartOfSpeechTypesService.get_mystem_pos_count())
    config.init_config_dependent_parameters()