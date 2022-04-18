from arekit.common.experiment.api.ctx_base import ExperimentContext
from arekit.common.experiment.data_type import DataType


class BaseIOUtils(object):
    """ Represents base experiment utils for input/output for:
        samples -- data that utilized for experiments;
        results -- evaluation of experiments.
    """

    def __init__(self, exp_ctx):
        assert(isinstance(exp_ctx, ExperimentContext))
        self._exp_ctx = exp_ctx
        self.__opinion_collection_provider = self._create_opinion_collection_provider()
        self.__opinion_collection_writer = self._create_opinion_collection_writer()

    def try_prepare(self):
        raise NotImplementedError()

    # region abstract methods

    def create_docs_stat_target(self):
        raise NotImplementedError()

    def create_samples_view(self, data_type):
        raise NotImplementedError()

    def create_opinions_view(self, data_type):
        raise NotImplementedError()

    def create_samples_writer(self):
        raise NotImplementedError()

    def create_opinions_writer(self):
        raise NotImplementedError()

    def create_samples_writer_target(self, data_type):
        raise NotImplementedError()

    def create_opinions_writer_target(self, data_type):
        raise NotImplementedError()

    def create_result_opinion_collection_target(self, doc_id, data_type, epoch_index):
        raise NotImplementedError()

    def _create_annotated_collection_target(self, doc_id, data_type, check_existance):
        raise NotImplementedError()

    def _create_opinion_collection_provider(self):
        raise NotImplementedError()

    def _create_opinion_collection_writer(self):
        raise NotImplementedError()

    # endregion

    # region public methods

    def balance_samples(self, data_type, balance):
        return balance and data_type == DataType.Train

    def create_opinion_collection_target(self, doc_id, data_type, check_existance=False):
        return self._create_annotated_collection_target(
            doc_id=doc_id,
            data_type=data_type,
            check_existance=check_existance)

    def write_opinion_collection(self, collection, labels_formatter, target):
        assert(target is not None)

        self.__opinion_collection_writer.serialize(
            target=target,
            collection=collection,
            encoding='utf-8',
            labels_formatter=labels_formatter)

    def read_opinion_collection(self, target, labels_formatter, create_collection_func,
                                error_on_non_supported=False):

        # Check existence of the target.
        if target is None:
            return None

        opinions = self.__opinion_collection_provider.iter_opinions(
            source=target,
            encoding='utf-8',
            labels_formatter=labels_formatter,
            error_on_non_supported=error_on_non_supported)

        return create_collection_func(opinions)

    # endregion
