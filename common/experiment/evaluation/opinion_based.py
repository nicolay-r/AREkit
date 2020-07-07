from arekit.common.experiment.formats.opinions import OpinionOperations
from arekit.common.model.eval.opinion_based import OpinionBasedModelEvaluator


class OpinionBasedExperimentEvaluator(OpinionBasedModelEvaluator):

    def __init__(self, evaluator, opin_ops):
        super(OpinionBasedExperimentEvaluator, self).__init__(evaluator=evaluator)
        assert(isinstance(opin_ops, OpinionOperations))
        self.__opin_ops = opin_ops

    def iter_opinion_collections_to_compare(self, data_type, doc_ids, epoch_index):
        assert(epoch_index == 0)

        return self.__opin_ops.iter_opinion_collections_to_compare(
            data_type=data_type,
            doc_ids=doc_ids,
            epoch_index=epoch_index)
