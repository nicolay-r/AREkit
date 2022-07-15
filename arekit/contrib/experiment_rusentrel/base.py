from arekit.common.experiment.api.ctx_base import ExperimentContext
from arekit.common.experiment.api.io_utils import BaseIOUtils
from arekit.common.experiment.api.ops_doc import DocumentOperations
from arekit.contrib.experiment_rusentrel.ops_opin import OpinionOperations


class BaseExperiment(object):

    def __init__(self, exp_ctx, exp_io, doc_ops, opin_ops):
        assert(isinstance(exp_ctx, ExperimentContext))
        assert(isinstance(exp_io, BaseIOUtils))
        assert(isinstance(doc_ops, DocumentOperations))
        assert(isinstance(opin_ops, OpinionOperations))
        self.__exp_ctx = exp_ctx
        self.__exp_io = exp_io
        self.__doc_ops = doc_ops
        self.__opin_ops = opin_ops

    @property
    def ExperimentContext(self):
        return self.__exp_ctx

    @property
    def ExperimentIO(self):
        return self.__exp_io

    @property
    def DocumentOperations(self):
        return self.__doc_ops

    @property
    def OpinionOperations(self):
        return self.__opin_ops
