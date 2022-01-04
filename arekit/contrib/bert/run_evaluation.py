import logging

from arekit.common.data import const
from arekit.common.data.views.linkages.multilabel import MultilableOpinionLinkagesView
from arekit.common.experiment.api.ctx_training import TrainingData
from arekit.common.experiment.api.enums import BaseDocumentTag
from arekit.common.experiment.engine import ExperimentEngine
from arekit.common.experiment.pipelines.opinion_collections import output_to_opinion_collections_pipeline
from arekit.common.labels.scaler import BaseLabelScaler
from arekit.common.labels.str_fmt import StringLabelsFormatter
from arekit.common.model.labeling.modes import LabelCalculationMode
from arekit.common.pipeline.context import PipelineContext
from arekit.common.pipeline.item_handle import HandleIterPipelineItem
from arekit.contrib.bert.callback import Callback
from arekit.contrib.bert.output.eval_helper import EvalHelper

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class LanguageModelExperimentEvaluator(ExperimentEngine):

    def __init__(self, experiment, data_type, eval_helper, max_epochs_count,
                 label_scaler, labels_formatter, eval_last_only=True, log_dir="./"):
        assert(isinstance(eval_helper, EvalHelper))
        assert(isinstance(max_epochs_count, int))
        assert(isinstance(eval_last_only, bool))
        assert(isinstance(label_scaler, BaseLabelScaler))
        assert(isinstance(labels_formatter, StringLabelsFormatter))
        assert(isinstance(log_dir, str))

        super(LanguageModelExperimentEvaluator, self).__init__(experiment=experiment)

        self.__data_type = data_type
        self.__eval_helper = eval_helper
        self.__max_epochs_count = max_epochs_count
        self.__eval_last_only = eval_last_only
        self.__labels_formatter = labels_formatter
        self.__label_scaler = label_scaler
        self.__log_dir = log_dir

    def _log_info(self, message, forced=False):
        assert(isinstance(message, str))

        if not self._experiment._do_log and not forced:
            return

        logger.info(message)

    def __run_pipeline(self, epoch_index, iter_index):
        exp_io = self._experiment.ExperimentIO
        exp_data = self._experiment.DataIO
        doc_ops = self._experiment.DocumentOperations

        cmp_doc_ids_set = set(doc_ops.iter_tagget_doc_ids(BaseDocumentTag.Compare))

        output_storage = exp_io.get_output_storage(
            epoch_index=epoch_index, iter_index=iter_index, eval_helper=self.__eval_helper)

        # We utilize google bert format, where every row
        # consist of label probabilities per every class
        linkages_view = MultilableOpinionLinkagesView(labels_scaler=self.__label_scaler,
                                                      storage=output_storage)

        ppl = output_to_opinion_collections_pipeline(
            iter_opinion_linkages_func=lambda doc_id: linkages_view.iter_opinion_linkages(
                doc_id=doc_id,
                opinions_view=exp_io.create_opinions_view(self.__data_type)),
            doc_ids_set=cmp_doc_ids_set,
            create_opinion_collection_func=self._experiment.OpinionOperations.create_opinion_collection,
            labels_scaler=self.__label_scaler,
            supported_labels=exp_data.SupportedCollectionLabels,
            label_calc_mode=LabelCalculationMode.AVERAGE)

        # Writing opinion collection.
        save_item = HandleIterPipelineItem(
            lambda data:
            exp_io.write_opinion_collection(
                collection=data[1],
                labels_formatter=self.__labels_formatter,
                target=exp_io.create_result_opinion_collection_target(
                    data_type=self.__data_type,
                    epoch_index=epoch_index,
                    doc_id=data[0])))

        # Executing pipeline.
        ppl.append(save_item)
        pipeline_ctx = PipelineContext({
            "src": set(output_storage.iter_column_values(column_name=const.DOC_ID))
        })
        ppl.run(pipeline_ctx)

        # iterate over the result.
        for _ in pipeline_ctx.provide("src"):
            pass

    def _handle_iteration(self, iter_index):
        exp_data = self._experiment.DataIO
        assert(isinstance(exp_data, TrainingData))

        # Setup callback.
        callback = exp_data.Callback
        assert(isinstance(callback, Callback))
        callback.set_iter_index(iter_index)

        if not self._experiment.ExperimentIO.try_prepare():
            return

        if callback.check_log_exists():
            self._log_info("Skipping [Log file already exist]")
            return

        with callback:
            for epoch_index in reversed(list(range(self.__max_epochs_count))):

                # Perform iteration related actions.
                self.__run_pipeline(epoch_index=epoch_index, iter_index=iter_index)

                # Evaluate.
                result = self._experiment.evaluate(data_type=self.__data_type, epoch_index=epoch_index)
                result.calculate()

                # Saving results.
                callback.write_results(result=result, data_type=self.__data_type, epoch_index=epoch_index)

                if self.__eval_last_only:
                    self._log_info("Evaluation done [Evaluating last only]")
                    return

    def _before_running(self):
        # Providing a root dir for logging.
        callback = self._experiment.DataIO.Callback
        callback.set_log_dir(self.__log_dir)
