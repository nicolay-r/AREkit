from arekit.common.experiment.formats.base import BaseExperiment
from arekit.common.model.eval.opinion_based import OpinionBasedModelEvaluator


class BERTModelEvaluator(OpinionBasedModelEvaluator):

    def __init__(self, evaluator, experiment):
        super(BERTModelEvaluator, self).__init__(evaluator=evaluator)
        assert(isinstance(experiment, BaseExperiment))
        self.__experiment = experiment

    def iter_opinion_collections_to_compare(self, data_type, doc_ids, epoch_index):
        assert(epoch_index == 0)

        return self.__experiment.iter_opinion_collections_to_compare(
            data_type=data_type,
            doc_ids=doc_ids,
            epoch_index=epoch_index)
