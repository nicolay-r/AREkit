from arekit.networks.multi.configurations.base import BaseMultiInstanceConfig


class AttSelfOverSentencesConfig(BaseMultiInstanceConfig):

    def __init__(self, context_config):
        super(AttSelfOverSentencesConfig, self).__init__(context_config)
