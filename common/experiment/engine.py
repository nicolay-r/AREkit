import os
import gc
import logging

from arekit.common.experiment.data_io import DataIO
from arekit.common.experiment.formats.base import BaseExperiment
from arekit.common.experiment.formats.cv_based.experiment import CVBasedExperiment
from arekit.networks.callback import Callback

from arekit.contrib.networks.context.configurations.base.base import DefaultNetworkConfig
from arekit.networks.data_handling.data import HandledData
from arekit.networks.feeding.bags.collection.base import BagsCollection
from arekit.networks.model import BaseTensorflowModel


class ExperimentEngine(object):

    @staticmethod
    def __run_cv_index(data_io,
                       experiment,
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

        data_io.CVFoldingAlgorithm.set_iteration_index(cv_index)

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
    def run_testing(full_model_name,
                    create_config,
                    create_network,
                    experiment_type,
                    bags_collection_type,
                    data_io,
                    cv_count=1,
                    common_callback_modification_func=None,
                    custom_config_modification_func=None,
                    common_config_modification_func=None):
        """
        :param bags_collection_type: BagsCollection
        :param data_io:
        :param full_model_name: unicode
            model name
        :param create_config: func
        :param create_network:
        :param experiment_type:
        :param cv_count: int, cv_count > 0
            1 -- considered a fixed train/test separation.
        :param common_callback_modification_func:
        :param common_config_modification_func:
            for all models
        :param custom_config_modification_func:
            for model
        """
        assert(issubclass(bags_collection_type, BagsCollection))
        assert(callable(experiment_type))
        assert(issubclass(experiment_type, BaseExperiment))
        assert(isinstance(full_model_name, unicode))
        assert(isinstance(data_io, DataIO))
        assert(isinstance(cv_count, int) and cv_count > 0)

        # Disable tensorflow logging
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        # Setup logging format
        stream_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s %(levelname)8s %(name)s | %(message)s')
        stream_handler.setFormatter(formatter)
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        logger.addHandler(stream_handler)

        # Log
        logger.info("Full-Model-Name: {}".format(full_model_name))

        # TODO. Refactor
        data_io.CVFoldingAlgorithm.set_cv_count(cv_count)
        data_io.set_model_name(full_model_name)
        data_io.ModelIO.set_model_name(value=full_model_name)

        # Creating experiment
        logger.info("Creating experiment ...")
        experiment = experiment_type(data_io=data_io,
                                     prepare_model_root=True)
        assert(isinstance(experiment, BaseExperiment))

        # Initialize data_io
        logger.info("Initialize data-io ...")
        for data_type in experiment.DocumentOperations.iter_suppoted_data_types():
            experiment.NeutralAnnotator.create_collection(data_type=data_type)

        callback = data_io.Callback
        callback.PredictVerbosePerFileStatistic = False

        for cv_index in range(data_io.CVFoldingAlgorithm.CVCount):
            logger.info("Running for cv-index={}".format(cv_index))
            ExperimentEngine.__run_cv_index(
                data_io=data_io,
                experiment=experiment,
                callback=callback,
                cv_index=cv_index,
                create_config_func=create_config,
                create_network_func=create_network,
                bags_collection_type=bags_collection_type,
                common_callback_modification_func=common_callback_modification_func,
                custom_config_modification_func=custom_config_modification_func,
                common_config_modification_func=common_config_modification_func)

