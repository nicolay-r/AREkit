from os.path import join

from arekit.contrib.networks.core.callback.np_writer import NpzDataWriter
from arekit.contrib.networks.core.callback_network import NetworkCallback


class HiddenStatesWriterCallback(NetworkCallback):

    def __init__(self, log_dir):
        super(HiddenStatesWriterCallback, self).__init__()

        self.__epochs_passed = 0
        self.__log_dir = log_dir
        self.__writer = NpzDataWriter()

    def __target_provider(self, name, epoch_index):
        return join(self.__log_dir, 'hparams_{name}_e{epoch}'.format(name=name, epoch=epoch_index))

    def on_epoch_finished(self, pipeline, operation_cancel):
        super(HiddenStatesWriterCallback, self).on_epoch_finished(pipeline=pipeline,
                                                                  operation_cancel=operation_cancel)

        self.__epochs_passed += 1

        names, values = self._model_ctx.get_hidden_parameters()

        assert(isinstance(names, list))
        assert(isinstance(values, list))
        assert(len(names) == len(values))

        for value_index, name in enumerate(names):
            self.__writer.write(data=values[value_index],
                                target=self.__target_provider(name=name, epoch_index=self.__epochs_passed))
