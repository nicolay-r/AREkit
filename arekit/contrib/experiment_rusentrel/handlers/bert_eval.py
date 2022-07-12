from os.path import join, exists

from arekit.common.data import const
from arekit.common.log_utils import logger
from arekit.contrib.experiment_rusentrel.bert.output_provider import GoogleBertOutputStorage
from arekit.contrib.experiment_rusentrel.handlers.eval_helper import EvalHelper
from arekit.contrib.utils.handlers.to_output import BaseOutputConverterIterationHandler


class ModelEvaluationIterationHandler(BaseOutputConverterIterationHandler):

    def __init__(self, exp_io, doc_ops, data_type, eval_helper, create_opinion_collection_func,
                 original_target_dir, max_epochs_count, label_scaler, labels_formatter):
        super(ModelEvaluationIterationHandler, self).__init__(
            exp_io=exp_io, doc_ops=doc_ops, data_type=data_type,
            create_opinion_collection_func=create_opinion_collection_func,
            label_scaler=label_scaler, labels_formatter=labels_formatter)
        self.__max_epochs_count = max_epochs_count
        self.__original_target_dir = original_target_dir
        self.__eval_helper = eval_helper

    def __create_target(self, doc_id, epoch_index):
        return self.__exp_io.create_result_opinion_collection_target(
            data_type=self._data_type, epoch_index=epoch_index, doc_id=doc_id)

    def _iter_output_and_target_pairs(self, iter_index):
        for epoch_index in reversed(list(range(self.__max_epochs_count))):
            output = self.get_output_storage(epoch_index=epoch_index,
                                             iter_index=iter_index,
                                             eval_helper=self.__eval_helper,
                                             original_target_dir=self.__original_target_dir)
            target_func = lambda doc_id: self.__create_target(doc_id=doc_id, epoch_index=epoch_index)
            return output, target_func

    @staticmethod
    def get_output_storage(epoch_index, iter_index, eval_helper, original_target_dir):
        assert(isinstance(eval_helper, EvalHelper))

        # NOTE: we wrap original dir using eval_helper implementation.
        # The latter allows us support a custom dir modifications while all the
        # required data stays unchanged in terms of paths.
        target_dir = eval_helper.get_results_dir(original_target_dir)

        result_filename = eval_helper.get_results_target(
            iter_index=iter_index,
            epoch_index=epoch_index)

        result_filepath = join(target_dir, result_filename)

        if not exists(result_filepath):
            logger.info("Result filepath was not found: {}".format(result_filepath))
            return None

        # Initialize storage.
        output_storage = GoogleBertOutputStorage.from_tsv(filepath=result_filepath, header=None)
        output_storage.apply_samples_view(
            row_ids=output_storage.iter_column_values(column_name=const.ID, dtype=str),
            doc_ids=output_storage.iter_column_values(column_name=const.DOC_ID, dtype=str))

        return output_storage

    def on_iteration(self, iter_index):
        if not self.__exp_io.try_prepare():
            return
        super(ModelEvaluationIterationHandler, self).on_iteration(iter_index)
