import logging
from os.path import exists, join

from arekit.common.data.views.output_multiple import MulticlassOutputView
from arekit.common.experiment.api.ctx_training import TrainingData
from arekit.common.experiment.api.enums import BaseDocumentTag
from arekit.common.experiment.engine import ExperimentEngine
from arekit.common.labels.scaler import BaseLabelScaler
from arekit.common.labels.str_fmt import StringLabelsFormatter
from arekit.common.model.labeling.modes import LabelCalculationMode
from arekit.common.model.labeling.single import SingleLabelsHelper
from arekit.common.opinions.base import Opinion
from arekit.common.utils import join_dir_with_subfolder_name
from arekit.contrib.bert.callback import Callback
from arekit.contrib.bert.output.eval_helper import EvalHelper
from arekit.contrib.bert.output.google_bert_provider import GoogleBertOutputStorage
from arekit.contrib.bert.utils import create_and_fill_opinion_collection

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class LanguageModelExperimentEvaluator(ExperimentEngine):

    def __init__(self, experiment, data_type, eval_helper, max_epochs_count,
                 label_scaler, labels_formatter, eval_last_only=True):
        assert(isinstance(eval_helper, EvalHelper))
        assert(isinstance(max_epochs_count, int))
        assert(isinstance(eval_last_only, bool))
        assert(isinstance(label_scaler, BaseLabelScaler))
        assert(isinstance(labels_formatter, StringLabelsFormatter))

        super(LanguageModelExperimentEvaluator, self).__init__(experiment=experiment)

        self.__data_type = data_type
        self.__eval_helper = eval_helper
        self.__max_epochs_count = max_epochs_count
        self.__eval_last_only = eval_last_only
        self.__labels_formatter = labels_formatter
        self.__label_scaler = label_scaler

    def _log_info(self, message, forced=False):
        assert(isinstance(message, str))

        if not self._experiment._do_log and not forced:
            return

        logger.info(message)

    def __get_target_dir(self):
        # NOTE: we wrap original dir using eval_helper implementation.
        # The latter allows us support a custom dir modifications while all the
        # required data stays unchanged in terms of paths.
        original_target_dir = self._experiment.ExperimentIO.get_target_dir()
        return self.__eval_helper.get_results_dir(original_target_dir)

    def _handle_iteration(self, iter_index):
        exp_io = self._experiment.ExperimentIO
        exp_data = self._experiment.DataIO
        assert(isinstance(exp_data, TrainingData))

        model_dir = self._experiment.ExperimentIO.get_target_dir()
        if not exists(model_dir):
            self._log_info("Model dir does not exist. Skipping")
            return

        # NOTE: since get_target_dir overrides the base implementation,
        # here we need to manually implement exp_dir (as in BaseIOUtils).
        # TODO. exp_dir creation should be outside of the run_evaluation script.
        # TODO. exp_dir creation should be outside of the run_evaluation script.
        # TODO. exp_dir creation should be outside of the run_evaluation script.
        # TODO. For global generalization purposes.
        # TODO. For global generalization purposes.
        # TODO. For global generalization purposes.
        exp_dir = join_dir_with_subfolder_name(
            subfolder_name=self._experiment.ExperimentIO.get_experiment_folder_name(),
            dir=self._experiment.ExperimentIO.get_experiment_sources_dir())
        if not exists(exp_dir):
            self._log_info("Experiment dir: {}".format(exp_dir))
            self._log_info("Experiment dir does not exist. Skipping")
            return

        # Setup callback.
        callback = exp_data.Callback
        assert(isinstance(callback, Callback))
        callback.set_iter_index(iter_index)

        # TODO. This should be removed as this is a part of the particular
        # experiment, not source!.
        cmp_doc_ids_set = set(self._experiment.DocumentOperations.iter_tagget_doc_ids(BaseDocumentTag.Compare))

        if callback.check_log_exists():
            self._log_info("Skipping [Log file already exist]")
            return

        with callback:
            for epoch_index in reversed(list(range(self.__max_epochs_count))):

                target_dir = self.__get_target_dir()
                result_filename = self.__eval_helper.get_results_filename(
                    iter_index=iter_index,
                    epoch_index=epoch_index)

                result_filepath = join(target_dir, result_filename)

                if not exists(result_filepath):
                    self._log_info("Result filepath was not found: {}".format(result_filepath))
                    continue

                # Forcely logging that evaluation will be started.
                self._log_info("\nStarting evaluation for: {}".format(result_filepath),
                               forced=True)

                # Initialize storage.
                storage = GoogleBertOutputStorage.from_tsv(filepath=result_filepath, header=None)
                storage.apply_samples_view(samples_view=exp_io.create_samples_view(self.__data_type))

                # We utilize google bert format, where every row
                # consist of label probabilities per every class
                output_view = MulticlassOutputView(
                    labels_scaler=self.__label_scaler,
                    storage=storage)

                # iterate opinion collections.
                collections_iter = output_view.iter_opinion_collections(
                    opinions_view=exp_io.create_opinions_view(self.__data_type),
                    keep_doc_id_func=lambda doc_id: doc_id in cmp_doc_ids_set,
                    to_collection_func=lambda linked_iter: self.__create_opinion_collection(
                        supported_labels=exp_data.SupportedCollectionLabels,
                        linked_iter=linked_iter))

                for doc_id, collection in collections_iter:

                    target = exp_io.create_result_opinion_collection_target(
                        data_type=self.__data_type,
                        epoch_index=epoch_index,
                        doc_id=doc_id)

                    exp_io.write_opinion_collection(
                        collection=collection,
                        labels_formatter=self.__labels_formatter,
                        target=target)

                # evaluate
                result = self._experiment.evaluate(data_type=self.__data_type,
                                                   epoch_index=epoch_index)
                result.calculate()

                # saving results.
                callback.write_results(result=result,
                                       data_type=self.__data_type,
                                       epoch_index=epoch_index)

                if self.__eval_last_only:
                    self._log_info("Evaluation done [Evaluating last only]")
                    return

    def _before_running(self):
        # Providing a root dir for logging.
        callback = self._experiment.DataIO.Callback
        callback.set_log_dir(self.__get_target_dir())

    def __create_opinion_collection(self, linked_iter, supported_labels):
        return create_and_fill_opinion_collection(
            create_opinion_collection=self._experiment.OpinionOperations.create_opinion_collection,
            data_linkage_iter=linked_iter,
            labels_helper=SingleLabelsHelper(self.__label_scaler),
            to_opinion_func=LanguageModelExperimentEvaluator.__create_labeled_opinion,
            label_calc_mode=LabelCalculationMode.AVERAGE,
            supported_labels=supported_labels)

    @staticmethod
    def __create_labeled_opinion(item, label):
        assert(isinstance(item, Opinion))
        return Opinion(source_value=item.SourceValue,
                       target_value=item.TargetValue,
                       sentiment=label)
