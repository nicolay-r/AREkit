from arekit.common.experiment.handlers.to_output import BaseOutputConverterIterationHandler


class ModelEvaluationIterationHandler(BaseOutputConverterIterationHandler):

    def __init__(self, exp_io, doc_ops, opin_ops, data_type, eval_helper,
                 max_epochs_count, label_scaler, labels_formatter):
        super(ModelEvaluationIterationHandler, self).__init__(
            exp_io=exp_io, doc_ops=doc_ops, opin_ops=opin_ops, data_type=data_type,
            label_scaler=label_scaler, labels_formatter=labels_formatter)
        self.__max_epochs_count = max_epochs_count
        self.__eval_helper = eval_helper

    def __create_target(self, doc_id, epoch_index):
        return self.__exp_io.create_result_opinion_collection_target(
            data_type=self._data_type, epoch_index=epoch_index, doc_id=doc_id)

    def _iter_output_and_target_pairs(self, iter_index):
        for epoch_index in reversed(list(range(self.__max_epochs_count))):
            output = self.__exp_io.get_output_storage(epoch_index=epoch_index,
                                                      iter_index=iter_index,
                                                      eval_helper=self.__eval_helper)
            target_func = lambda doc_id: self.__create_target(doc_id=doc_id, epoch_index=epoch_index)
            return output, target_func

    def on_iteration(self, iter_index):
        if not self.__exp_io.try_prepare():
            return
        super(ModelEvaluationIterationHandler, self).on_iteration(iter_index)
