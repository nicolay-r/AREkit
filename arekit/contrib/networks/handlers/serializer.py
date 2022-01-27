from arekit.common.experiment.api.ctx_serialization import ExperimentSerializationContext
from arekit.common.experiment.api.ops_doc import DocumentOperations
from arekit.common.experiment.api.ops_opin import OpinionOperations
from arekit.common.experiment.handler import ExperimentIterationHandler
from arekit.contrib.experiment_rusentrel.model_io.tf_networks import RuSentRelExperimentNetworkIOUtils
from arekit.contrib.networks.core.input.helper import NetworkInputHelper


class NetworksInputSerializerExperimentIteration(ExperimentIterationHandler):

    def __init__(self, exp_ctx, exp_io, doc_ops, opin_ops, force_serialize,
                 value_to_group_id_func, balance, skip_folder_if_exists):
        assert(callable(value_to_group_id_func))
        assert(isinstance(exp_ctx, ExperimentSerializationContext))
        assert(isinstance(exp_io, RuSentRelExperimentNetworkIOUtils))
        assert(isinstance(doc_ops, DocumentOperations))
        assert(isinstance(opin_ops, OpinionOperations))
        assert(isinstance(force_serialize, bool))
        assert(isinstance(balance, bool))
        super(NetworksInputSerializerExperimentIteration, self).__init__()

        self.__exp_ctx = exp_ctx
        self.__exp_io = exp_io
        self.__doc_ops = doc_ops
        self.__opin_ops = opin_ops
        self.__force_serialize = force_serialize
        self.__skip_folder_if_exists = skip_folder_if_exists
        self.__value_to_group_id_func = value_to_group_id_func
        self.__balance = balance

    # region protected methods

    def on_iteration(self, iter_index):
        targets_existed = self.__exp_io.check_targets_existed(
            data_types_iter=self.__exp_ctx.DataFolding.iter_supported_data_types())

        if targets_existed and not self.__force_serialize:
            return

        # Perform data serialization.
        NetworkInputHelper.prepare(exp_io=self.__exp_io,
                                   exp_ctx=self.__exp_ctx,
                                   doc_ops=self.__doc_ops,
                                   opin_ops=self.__opin_ops,
                                   terms_per_context=self.__exp_ctx.TermsPerContext,
                                   balance=self.__balance,
                                   value_to_group_id_func=self.__value_to_group_id_func)

    # endregion