from arekit.common.experiment.api.ctx_base import ExperimentContext


class BaseIOUtils(object):
    """ Represents base experiment utils for input/output for:
        samples -- data that utilized for experiments;
        results -- evaluation of experiments.
    """

    def __init__(self, exp_ctx):
        assert(isinstance(exp_ctx, ExperimentContext))
        self._exp_ctx = exp_ctx

    # region abstract methods

    def try_prepare(self):
        raise NotImplementedError()

    def get_target_dir(self):
        raise NotImplementedError()

    def create_samples_view(self, data_type, data_folding):
        raise NotImplementedError()

    def create_opinions_view(self, target):
        raise NotImplementedError()

    def create_samples_writer(self):
        raise NotImplementedError()

    def create_opinions_writer(self):
        raise NotImplementedError()

    def create_samples_writer_target(self, data_type, data_folding):
        raise NotImplementedError()

    def create_opinions_writer_target(self, data_type, data_folding):
        raise NotImplementedError()

    # endregion
