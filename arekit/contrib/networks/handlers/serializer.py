from arekit.common.experiment.api.ctx_serialization import ExperimentSerializationContext
from arekit.common.experiment.api.ops_doc import DocumentOperations
from arekit.common.experiment.api.ops_opin import OpinionOperations
from arekit.common.experiment.handler import ExperimentIterationHandler
from arekit.common.text.parser import BaseTextParser
from arekit.contrib.experiment_rusentrel.model_io.tf_networks import RuSentRelExperimentNetworkIOUtils
from arekit.contrib.networks.core.input.helper import NetworkInputHelper


class NetworksInputSerializerExperimentIteration(ExperimentIterationHandler):

    def __init__(self, exp_ctx, exp_io, doc_ops, opin_ops, value_to_group_id_func, text_parser, balance):
        assert(callable(value_to_group_id_func))
        assert(isinstance(exp_ctx, ExperimentSerializationContext))
        assert(isinstance(exp_io, RuSentRelExperimentNetworkIOUtils))
        assert(isinstance(doc_ops, DocumentOperations))
        assert(isinstance(opin_ops, OpinionOperations))
        assert(isinstance(text_parser, BaseTextParser))
        assert(isinstance(balance, bool))
        super(NetworksInputSerializerExperimentIteration, self).__init__()

        self.__exp_ctx = exp_ctx
        self.__exp_io = exp_io
        self.__doc_ops = doc_ops
        self.__opin_ops = opin_ops
        self.__value_to_group_id_func = value_to_group_id_func
        self.__balance = balance
        self.__text_parser = text_parser

    # region protected methods

    def on_iteration(self, iter_index):

        # Perform data serialization.
        NetworkInputHelper.prepare(exp_io=self.__exp_io,
                                   exp_ctx=self.__exp_ctx,
                                   doc_ops=self.__doc_ops,
                                   opin_ops=self.__opin_ops,
                                   terms_per_context=self.__exp_ctx.TermsPerContext,
                                   balance=self.__balance,
                                   value_to_group_id_func=self.__value_to_group_id_func,
                                   text_parser=self.__text_parser)

    # endregion
