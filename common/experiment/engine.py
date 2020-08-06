import os
import gc
import logging

from arekit.common.experiment.formats.base import BaseExperiment
from arekit.common.experiment.formats.cv_based.experiment import CVBasedExperiment

from arekit.contrib.networks.context.configurations.base.base import DefaultNetworkConfig
from arekit.contrib.networks.core.callback.base import Callback
from arekit.contrib.networks.core.data_handling.data import HandledData
from arekit.contrib.networks.core.feeding.bags.collection.base import BagsCollection
from arekit.contrib.networks.core.model import BaseTensorflowModel


class ExperimentEngine(object):

    @staticmethod
    def __run_cv_index(experiment,
                       bags_collection_type,
                       create_config_func,
                       create_network_func,
                       callback,
                       cv_index,
                       common_callback_modification_func=None,
                       custom_config_modification_func=None,
                       common_config_modification_func=None):
        """
        Run single CV-index experiment.
        """
        assert(isinstance(callback, Callback))
        assert(isinstance(experiment, CVBasedExperiment))
        assert(issubclass(bags_collection_type, BagsCollection))
        assert(callable(create_config_func))
        assert(callable(create_network_func))
        assert(callable(common_callback_modification_func) or common_callback_modification_func is None)
        assert(callable(common_config_modification_func) or common_config_modification_func is None)
        assert(callable(custom_config_modification_func) or custom_config_modification_func is None)
        assert(isinstance(cv_index, int))

        # Initialize config
        config = create_config_func()
        assert(isinstance(config, DefaultNetworkConfig))

        # Initialize network
        network = create_network_func()

        # Setup config
        config.modify_classes_count(value=experiment.DataIO.LabelsScaler.classes_count())

        if common_config_modification_func is not None:
            common_config_modification_func(config=config)
        if custom_config_modification_func is not None:
            custom_config_modification_func(config)

        # Setup callback
        if common_callback_modification_func is not None:
            common_callback_modification_func(callback)
        callback.reset_experiment_dependent_parameters()
        callback.set_cv_index(cv_index)
        callback.set_experiment(experiment)

        # Perform data handling.
        handled_data = HandledData.initialize_from_experiment(
            experiment=experiment,
            config=config,
            bags_collection_type=bags_collection_type)

        # Initialize model
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

        del config
        del network
        del model

        gc.collect()

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

        # Setup logging format
        stream_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s %(levelname)8s %(name)s | %(message)s')
        stream_handler.setFormatter(formatter)
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        logger.addHandler(stream_handler)

        # Initializing annotator
        logger.info("Initializing neutral annotator ...")
        experiment.initialize_neutral_annotator()

        # Setup model root

        # Initialize data_io
        logger.info("Initialize data-io ...")
        for data_type in experiment.DocumentOperations.iter_suppoted_data_types():
            experiment.NeutralAnnotator.create_collection(data_type=data_type)

        callback = experiment.DataIO.Callback
        callback.PredictVerbosePerFileStatistic = False

        for cv_index in range(experiment.DataIO.CVFoldingAlgorithm.CVCount):
            logger.info("Running for cv-index={}".format(cv_index))

            experiment.DataIO.CVFoldingAlgorithm.set_iteration_index(cv_index)

            ExperimentEngine.__run_cv_index(
                experiment=experiment,
                callback=callback,
                cv_index=cv_index,
                create_config_func=create_config,
                create_network_func=create_network,
                bags_collection_type=bags_collection_type,
                common_callback_modification_func=common_callback_modification_func,
                custom_config_modification_func=custom_config_modification_func,
                common_config_modification_func=common_config_modification_func)

