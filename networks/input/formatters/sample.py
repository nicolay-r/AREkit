from arekit.common.experiment.input.formatters.sample.base import BaseSampleFormatter


class NetworkSample(BaseSampleFormatter):

    def __init__(self, data_type, label_provider, text_provider, write_embedding_pair_func):
        assert(callable(write_embedding_pair_func))

        super(NetworkSample, self).__init__(data_type=data_type,
                                            label_provider=label_provider,
                                            text_provider=text_provider)

        # TODO. Utilize this parameter.
        self.__write_embedding_pair_func = write_embedding_pair_func

