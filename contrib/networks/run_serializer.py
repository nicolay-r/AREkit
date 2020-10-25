from arekit.common.experiment.engine.cv_based import CVBasedExperimentEngine
from arekit.common.experiment.engine.utils import mark_dir_for_serialization
from arekit.common.experiment.neutral.run import perform_neutral_annotation
from arekit.contrib.networks.core.data_handling.data import HandledData
from arekit.contrib.networks.core.io_utils import NetworkIOUtils
from arekit.contrib.networks.init_config import initialize_config


class NetworksExperimentInputSerializer(CVBasedExperimentEngine):

    def __init__(self, experiment, create_config, skip_folder_if_exists, io_utils=NetworkIOUtils):
        assert(issubclass(io_utils, NetworkIOUtils))
        assert(issubclass(skip_folder_if_exists, bool))
        assert(callable(create_config))

        super(NetworksExperimentInputSerializer, self).__init__(experiment)

        self.__config = None
        self.__io_utils = io_utils
        self.__skip_folder_if_exists = skip_folder_if_exists
        self.__create_config = create_config

    # region protected methods

    def _handle_cv_index(self, cv_index):

        # Performing data serialization.
        if not HandledData.need_serialize(self._experiment):
            return

        # Perform data serialization.
        HandledData.serialize_from_experiment(experiment=self._experiment,
                                              config=self.__config)

    def _before_running(self):
        # Mark the directory as selected for serialization process.
        mark_dir_for_serialization(target_dir=self.__io_utils.get_target_dir(self._experiment),
                                   logger=self._logger,
                                   skip_if_folder_exists=self.__skip_folder_if_exists)

        # Perform neutral annotation.
        perform_neutral_annotation(experiment=self._experiment,
                                   logger=self._logger)

        # Create config.
        self.__config = initialize_config(create_config_func=self.__create_config,
                                          classes_count=self._experiment.DataIO.LabelsScaler.classes_count())

    #  endregion