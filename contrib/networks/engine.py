import os
import gc
import logging

from arekit.common.experiment.formats.base import BaseExperiment
from arekit.common.experiment.formats.cv_based.experiment import CVBasedExperiment

from arekit.contrib.networks.context.configurations.base.base import DefaultNetworkConfig
from arekit.contrib.networks.core.callback.base import Callback
from arekit.contrib.networks.core.data_handling.data import HandledData
from arekit.contrib.networks.core.feeding.bags.collection.base import BagsCollection
from arekit.contrib.networks.core.io_utils import NetworkIOUtils
from arekit.contrib.networks.core.model import BaseTensorflowModel


class ExperimentEngine(object):

    # region private methods

    @staticmethod
    def __run_cv_index(experiment,
                       bags_collection_type,
                       create_network_func,
                       config,
                       callback,
                       cv_index):
        """
        Run single CV-index experiment.
        """
        assert(isinstance(config, DefaultNetworkConfig))
        assert(isinstance(callback, Callback))
        assert(isinstance(experiment, CVBasedExperiment))
        assert(issubclass(bags_collection_type, BagsCollection))
        assert(callable(create_network_func))
        assert(isinstance(cv_index, int))

        # Perform data reading.
        handled_data = HandledData.create_empty()
        handled_data.perform_reading_and_initialization(
            experiment=experiment,
            bags_collection_type=bags_collection_type,
            config=config)

        # Setup callback
        callback.reset_experiment_dependent_parameters()
        callback.set_cv_index(cv_index)
        callback.set_experiment(experiment)

        # Initialize network and model
        network = create_network_func()
        model = BaseTensorflowModel(network=network,
                                    config=config,
                                    handled_data=handled_data,
                                    bags_collection_type=bags_collection_type,
                                    callback=callback,
                                    nn_io=experiment.DataIO.ModelIO,
                                    label_scaler=experiment.DataIO.LabelsScaler,
                                    evaluator=experiment.DataIO.Evaluator)

        # Run model
        model.run_training(load_model=False, epochs_count=callback.Epochs)

        del network
        del model

        gc.collect()

    @staticmethod
    def __setup_logger():
        stream_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s %(levelname)8s %(name)s | %(message)s')
        stream_handler.setFormatter(formatter)
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        logger.addHandler(stream_handler)

        return logger

    @staticmethod
    def __sutup_experiment(logger, experiment):
        assert(isinstance(experiment, BaseExperiment))
        # Initializing annotator
        logger.info("Initializing neutral annotator ...")
        experiment.initialize_neutral_annotator()

        # Initialize data_io
        logger.info("Initialize data-io ...")
        for data_type in experiment.DocumentOperations.iter_suppoted_data_types():
            experiment.NeutralAnnotator.create_collection(data_type=data_type)

    @staticmethod
    def __serialize_experiment_data(logger, experiment, config):
        # Performing data serialization.

        for cv_index in ExperimentEngine.__iter_cv_index(experiment):
            logger.info("Serializing data for cv-index={}".format(cv_index))

            if not HandledData.need_serialize(experiment):
                continue

            # Perform data serialization.
            HandledData.serialize_from_experiment(
                experiment=experiment,
                config=config)

    @staticmethod
    def __iter_cv_index(experiment):
        for cv_index in range(experiment.DataIO.CVFoldingAlgorithm.CVCount):
            experiment.DataIO.CVFoldingAlgorithm.set_iteration_index(cv_index)
            yield cv_index

    @staticmethod
    def __create_config(create_config_func,
                        classes_count,
                        custom_config_modification_func=None,
                        common_config_modification_func=None):
        assert(isinstance(classes_count, int))
        assert(callable(create_config_func))
        assert(callable(common_config_modification_func) or common_config_modification_func is None)
        assert(callable(custom_config_modification_func) or custom_config_modification_func is None)

        # Initialize config
        config = create_config_func()

        assert(isinstance(config, DefaultNetworkConfig))

        # Setup config
        config.modify_classes_count(value=classes_count)

        if common_config_modification_func is not None:
            common_config_modification_func(config=config)
        if custom_config_modification_func is not None:
            custom_config_modification_func(config)

        return config

    # endregion

    # region public methods

    @staticmethod
    def run_serialization(logger, experiment, create_config, skip_if_folder_exists):
        assert(isinstance(experiment, BaseExperiment))
        assert(callable(create_config))
        target_dir = NetworkIOUtils.get_target_dir(experiment)
        target_file = os.path.join(target_dir, 'lock.txt')
        if os.path.exists(target_file) and skip_if_folder_exists:
            logger.info("TARGET DIR EXISTS: {}".format(target_dir))
            return
        else:
            open(target_file, 'a').close()

        ExperimentEngine.__sutup_experiment(experiment=experiment,
                                            logger=logger)

        # Create config.
        config = ExperimentEngine.__create_config(
            create_config_func=create_config,
            classes_count=experiment.DataIO.LabelsScaler.classes_count())

        # Running serialization.
        ExperimentEngine.__serialize_experiment_data(logger=logger,
                                                     experiment=experiment,
                                                     config=config)

    @staticmethod
    def run_testing(create_config,
                    create_network,
                    experiment,
                    bags_collection_type,
                    common_callback_modification_func=None,
                    custom_config_modification_func=None,
                    common_config_modification_func=None):
        assert(issubclass(bags_collection_type, BagsCollection))
        assert(isinstance(experiment, BaseExperiment))

        # Disable tensorflow logging
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        logger = ExperimentEngine.__setup_logger()

        ExperimentEngine.__sutup_experiment(experiment=experiment,
                                            logger=logger)

        # Create config
        config = ExperimentEngine.__create_config(
            create_config_func=create_config,
            classes_count=experiment.DataIO.LabelsScaler.classes_count(),
            custom_config_modification_func=custom_config_modification_func,
            common_config_modification_func=common_config_modification_func)

        ExperimentEngine.__serialize_experiment_data(logger=logger,
                                                     experiment=experiment,
                                                     config=config)

        # Setup callback
        callback = experiment.DataIO.Callback
        callback.PredictVerbosePerFileStatistic = False
        if common_callback_modification_func is not None:
            common_callback_modification_func(callback)

        # Performing data reading and running experiments.
        for cv_index in ExperimentEngine.__iter_cv_index(experiment):
            logger.info("Running for cv-index={}".format(cv_index))

            ExperimentEngine.__run_cv_index(
                experiment=experiment,
                callback=callback,
                cv_index=cv_index,
                config=config,
                create_network_func=create_network,
                bags_collection_type=bags_collection_type)

    # endregion