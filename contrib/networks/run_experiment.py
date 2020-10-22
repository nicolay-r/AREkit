import os
import gc

from arekit.common.experiment.engine import BaseExperimentEngine
from arekit.common.experiment.formats.base import BaseExperiment
from arekit.common.experiment.formats.cv_based.experiment import CVBasedExperiment

from arekit.contrib.networks.context.configurations.base.base import DefaultNetworkConfig
from arekit.contrib.networks.core.callback.base import Callback
from arekit.contrib.networks.core.data_handling.data import HandledData
from arekit.contrib.networks.core.feeding.bags.collection.base import BagsCollection
from arekit.contrib.networks.core.model import BaseTensorflowModel
from arekit.contrib.networks.init_config import initialize_config


class NetworksExperimentEngine(BaseExperimentEngine):

    # region private methods

    @staticmethod
    def __run_cv_index(experiment,
                       bags_collection_type,
                       create_network_func,
                       config,
                       callback,
                       cv_index,
                       load_model):
        """
        Run single CV-index experiment.
        """
        assert(isinstance(config, DefaultNetworkConfig))
        assert(isinstance(callback, Callback))
        assert(isinstance(experiment, CVBasedExperiment))
        assert(issubclass(bags_collection_type, BagsCollection))
        assert(callable(create_network_func))
        assert(isinstance(cv_index, int))
        assert(isinstance(load_model, bool))

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

    # endregion

    @staticmethod
    def run_testing(create_config,
                    create_network,
                    experiment,
                    bags_collection_type,
                    load_model=False,
                    common_callback_modification_func=None,
                    custom_config_modification_func=None,
                    common_config_modification_func=None):
        assert(issubclass(bags_collection_type, BagsCollection))
        assert(isinstance(experiment, BaseExperiment))
        assert(isinstance(load_model, bool))

        # Disable tensorflow logging
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        logger = BaseExperimentEngine._setup_logger()

        # Create config
        config = initialize_config(
            create_config_func=create_config,
            classes_count=experiment.DataIO.LabelsScaler.classes_count(),
            custom_config_modification_func=custom_config_modification_func,
            common_config_modification_func=common_config_modification_func)

        # Setup callback
        callback = experiment.DataIO.Callback
        callback.PredictVerbosePerFileStatistic = False
        if common_callback_modification_func is not None:
            common_callback_modification_func(callback)

        # Performing data reading and running experiments.
        for cv_index in BaseExperimentEngine._iter_cv_index(experiment):
            logger.info("Running for cv-index={}".format(cv_index))

            NetworksExperimentEngine.__run_cv_index(
                experiment=experiment,
                callback=callback,
                cv_index=cv_index,
                config=config,
                create_network_func=create_network,
                bags_collection_type=bags_collection_type,
                load_model=load_model)