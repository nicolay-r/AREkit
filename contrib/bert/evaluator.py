from arekit.contrib.experiments.experiment_io import BaseExperimentNeuralNetworkIO
from arekit.networks.eval.opinion_based import OpinionBasedModelEvaluator


class BERTModelEvaluator(OpinionBasedModelEvaluator):

    def __init__(self, evaluator, experiment_io):
        super(BERTModelEvaluator, self).__init__(evaluator=evaluator)
        assert(isinstance(experiment_io, BaseExperimentNeuralNetworkIO))
        self.__io = experiment_io

    def iter_opinion_collections_to_compare(self, data_type, doc_ids, epoch_index):
        assert(epoch_index == 0)

        return self.__io.iter_opinion_collections_to_compare(
            data_type=data_type,
            doc_ids=doc_ids,
            epoch_index=epoch_index)
