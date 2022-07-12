from arekit.common.experiment.api.base import BaseExperiment
from arekit.contrib.experiment_rusentrel.ops_opin import OpinionOperations


class CustomExperiment(BaseExperiment):

    def __init__(self, exp_ctx, exp_io, doc_ops, opin_ops):
        assert(isinstance(opin_ops, OpinionOperations))
        super(CustomExperiment, self).__init__(exp_ctx=exp_ctx, exp_io=exp_io, doc_ops=doc_ops)
        self.__opin_ops = opin_ops

    @property
    def OpinionOperations(self):
        return self.__opin_ops
