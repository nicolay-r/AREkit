from arekit.common.experiment.engine import BaseExperimentEngine
from arekit.common.experiment.formats.base import BaseExperiment
from arekit.contrib.networks.core.data_handling.data import HandledData
from arekit.contrib.networks.core.io_utils import NetworkIOUtils
from arekit.contrib.networks.init_config import initialize_config


class NetworksExperimentInputSerializer(BaseExperimentEngine):

    # region private methods

    @staticmethod
    def __serialize_experiment_data(logger, experiment, config):
        # Performing data serialization.

        for cv_index in BaseExperimentEngine._iter_cv_index(experiment):
            logger.info("Serializing data for cv-index={}".format(cv_index))

            if not HandledData.need_serialize(experiment):
                continue

            # Perform data serialization.
            HandledData.serialize_from_experiment(experiment=experiment,
                                                  config=config)

    # endregion

    @staticmethod
    def run_serialization(logger, experiment, create_config, skip_if_folder_exists, io_utils=NetworkIOUtils):
        assert(isinstance(experiment, BaseExperiment))
        assert(issubclass(io_utils, NetworkIOUtils))
        assert(callable(create_config))

        # Mark the directory as selected for serialization process.
        BaseExperimentEngine._mark_dir_for_serialization(experiment=experiment,
                                                         logger=logger,
                                                         io_utils=io_utils,
                                                         skip_if_folder_exists=skip_if_folder_exists)

        # Perform neutral annotation.
        BaseExperimentEngine._perform_neutral_annotation(experiment=experiment,
                                                         logger=logger)

        # Create config.
        config = initialize_config(create_config_func=create_config,
                                   classes_count=experiment.DataIO.LabelsScaler.classes_count())

        # Running serialization.
        NetworksExperimentInputSerializer.__serialize_experiment_data(logger=logger,
                                                                      experiment=experiment,
                                                                      config=config)
