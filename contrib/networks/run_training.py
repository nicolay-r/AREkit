import os
import gc

from arekit.common.experiment.engine import CVBasedExperimentEngine
from arekit.contrib.networks.core.data_handling.data import HandledData
from arekit.contrib.networks.core.feeding.bags.collection.base import BagsCollection
from arekit.contrib.networks.core.model import BaseTensorflowModel
from arekit.contrib.networks.init_config import initialize_config


class NetworksTrainingEngine(CVBasedExperimentEngine):

    def __init__(self, create_config, bags_collection_type, experiment, load_model,
                 create_network_func,
                 common_callback_modification_func=None,
                 custom_config_modification_func=None,
                 common_config_modification_func=None):
        assert(callable(create_network_func))
        assert(callable(create_config))
        assert(issubclass(bags_collection_type, BagsCollection))
        assert(isinstance(load_model, bool))

        super(NetworksTrainingEngine, self).__init__(experiment)

        self.__config = None

        self.__create_config = create_config
        self.__create_network_func = create_network_func
        self.__bags_collection_type = bags_collection_type
        self.__load_model = load_model

        self.__common_callback_modification_func = common_callback_modification_func
        self.__custom_config_modification_func = custom_config_modification_func
        self.__common_config_modification_func = common_config_modification_func

    # region protected methods

    def _handle_cv_index(self, cv_index):
        """ Run single CV-index experiment.
        """
        assert(isinstance(cv_index, int))
        assert(self.__config is not None)

        # Perform data reading.
        handled_data = HandledData.create_empty()
        handled_data.perform_reading_and_initialization(experiment=self._experiment,
                                                        bags_collection_type=self.__bags_collection_type,
                                                        config=self.__config)

        # Setup callback
        callback = self._experiment.DataIO.Callback
        callback.reset_experiment_dependent_parameters()
        callback.set_cv_index(cv_index)
        callback.set_experiment(self._experiment)

        # Initialize network and model
        network = self.__create_network_func()
        model = BaseTensorflowModel(network=network,
                                    config=self.__config,
                                    handled_data=handled_data,
                                    bags_collection_type=self.__bags_collection_type,
                                    callback=callback,
                                    nn_io=self._experiment.DataIO.ModelIO,
                                    label_scaler=self._experiment.DataIO.LabelsScaler,
                                    evaluator=self._experiment.DataIO.Evaluator)

        # Run model
        model.run_training(load_model=False,
                           epochs_count=callback.Epochs)

        del network
        del model

        gc.collect()

    def _before_running(self):

        # Disable tensorflow logging
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        # Create config
        self.__config = initialize_config(create_config_func=self.__create_config,
                                          classes_count=self._experiment.DataIO.LabelsScaler.classes_count(),
                                          custom_config_modification_func=self.__custom_config_modification_func,
                                          common_config_modification_func=self.__common_config_modification_func)

        # Init callback
        callback = self._experiment.DataIO.Callback
        callback.PredictVerbosePerFileStatistic = False
        if self.__common_callback_modification_func is not None:
            self.__common_callback_modification_func(callback)

    # endregion