from arekit.common.experiment.api.ctx_serialization import SerializationData
from arekit.common.experiment.engine import ExperimentEngine
from arekit.contrib.networks.core.input.helper import NetworkInputHelper


# TODO. 262. Refactor as handler (weird inheritance, limits capabilities).
class NetworksExperimentInputSerializer(ExperimentEngine):

    def __init__(self, experiment, force_serialize, value_to_group_id_func, balance, skip_folder_if_exists):
        assert(callable(value_to_group_id_func))
        assert(isinstance(force_serialize, bool))
        assert(isinstance(balance, bool))

        super(NetworksExperimentInputSerializer, self).__init__(experiment)

        self.__force_serialize = force_serialize
        self.__skip_folder_if_exists = skip_folder_if_exists
        self.__value_to_group_id_func = value_to_group_id_func
        self.__balance = balance

    # region protected methods

    def _handle_iteration(self, it_index):
        assert(isinstance(self._experiment.DataIO, SerializationData))

        targets_existed = self._experiment.ExperimentIO.check_targets_existed(
            data_types_iter=self._experiment.DocumentOperations.DataFolding.iter_supported_data_types())

        if targets_existed and not self.__force_serialize:
            return

        # Perform data serialization.
        NetworkInputHelper.prepare(experiment=self._experiment,
                                   terms_per_context=self._experiment.DataIO.TermsPerContext,
                                   balance=self.__balance,
                                   value_to_group_id_func=self.__value_to_group_id_func)

    # endregion
