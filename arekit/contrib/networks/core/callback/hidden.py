from os.path import join

from arekit.contrib.networks.core.base_writer import BaseWriter
from arekit.contrib.networks.core.callback.base import NetworkCallback


class HiddenStatesWriterCallback(NetworkCallback):

    def __init__(self, log_dir, writer):
        assert(isinstance(writer, BaseWriter))
        super(HiddenStatesWriterCallback, self).__init__()

        self.__epochs_passed = 0
        self.__log_dir = log_dir
        self.__writer = writer

    def __target_provider(self, name, epoch_index):
        return join(self.__log_dir, 'hparams_{name}_e{epoch}'.format(name=name, epoch=epoch_index))

    def on_epoch_finished(self, pipeline, operation_cancel):
        super(HiddenStatesWriterCallback, self).on_epoch_finished(pipeline=pipeline,
                                                                  operation_cancel=operation_cancel)

        self.__epochs_passed += 1

        if len(pipeline) == 0:
            return

        model_ctx = pipeline[0].ModelContext
        names, tensors = map(list, zip(*model_ctx.Network.iter_hidden_parameters()))
        values = model_ctx.Session.run(tensors)

        assert(isinstance(names, list))
        assert(isinstance(values, list))
        assert(len(names) == len(values))

        for value_index, name in enumerate(names):
            self.__writer.write(data=values[value_index],
                                target=self.__target_provider(name=name, epoch_index=self.__epochs_passed))
