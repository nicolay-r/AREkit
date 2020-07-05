from arekit.common.experiment.input.formatters.sample.base import BaseSampleFormatter


class NetworkSample(BaseSampleFormatter):

    def __init__(self, data_type, label_provider, text_provider):
        super(NetworkSample, self).__init__(data_type=data_type,
                                            label_provider=label_provider,
                                            text_provider=text_provider)

