from arekit.common.experiment.api.ctx_base import ExperimentContext
from arekit.common.experiment.api.io_utils import BaseIOUtils
from arekit.common.experiment.api.ops_doc import DocumentOperations
from arekit.common.experiment.api.ops_opin import OpinionOperations


class BaseExperiment(object):

    def __init__(self, exp_ctx, exp_io, opin_ops, doc_ops):
        assert(isinstance(exp_ctx, ExperimentContext))
        assert(isinstance(exp_io, BaseIOUtils))
        assert(isinstance(opin_ops, OpinionOperations))
        assert(isinstance(doc_ops, DocumentOperations))
        self.__exp_ctx = exp_ctx
        self.__exp_io = exp_io
        self.__opin_ops = opin_ops
        self.__doc_ops = doc_ops

    # region Properties

    @property
    def ExperimentContext(self):
        return self.__exp_ctx

    @property
    def ExperimentIO(self):
        return self.__exp_io

    @property
    def OpinionOperations(self):
        return self.__opin_ops

    @property
    def DocumentOperations(self):
        return self.__doc_ops

    # endregion
