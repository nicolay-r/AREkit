from arekit.contrib.experiments.nn_io.base import BaseExperimentNeuralNetworkIO
from arekit.networks.eval.opinion_based import OpinionBasedModelEvaluator


class CustomOpinionBasedModelEvaluator(OpinionBasedModelEvaluator):

    def __init__(self, evaluator, nn_io):
        assert(isinstance(nn_io, BaseExperimentNeuralNetworkIO))
        super(CustomOpinionBasedModelEvaluator, self).__init__(evaluator=evaluator)
        self.__nn_io = nn_io

    def iter_opinion_collections_to_compare(self, data_type, doc_ids, epoch_index):
        return self.__nn_io.iter_opinion_collections_to_compare(
            data_type=data_type,
            doc_ids=doc_ids,
            epoch_index=epoch_index)

