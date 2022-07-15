from arekit.common.experiment.api.ctx_base import ExperimentContext


class BaseIOUtils(object):
    """ Represents base experiment utils for input/output for:
        samples -- data that utilized for experiments;
        results -- evaluation of experiments.
    """

    def __init__(self, exp_ctx):
        assert(isinstance(exp_ctx, ExperimentContext))
        self._exp_ctx = exp_ctx
        self.__opinion_collection_writer = self._create_opinion_collection_writer()

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

    def _create_opinion_collection_writer(self):
        raise NotImplementedError()

    # endregion

    # region public methods

    def write_opinion_collection(self, collection, labels_formatter, target):
        assert(target is not None)

        self.__opinion_collection_writer.serialize(
            target=target,
            collection=collection,
            encoding='utf-8',
            labels_formatter=labels_formatter)

    # endregion
