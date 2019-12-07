from arekit.networks.multi.configuration.base import BaseMultiInstanceConfig


class AttHiddenOverSentencesConfig(BaseMultiInstanceConfig):

    def __init__(self, context_config):
        super(AttHiddenOverSentencesConfig, self).__init__(context_config)
