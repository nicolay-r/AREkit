import os
import gc

from arekit.common.experiment.engine.cv_based import ExperimentEngine
from arekit.common.experiment.engine.utils import rm_dir_contents
from arekit.contrib.networks.context.configurations.base.base import DefaultNetworkConfig
from arekit.contrib.networks.core.data_handling.data import HandledData
from arekit.contrib.networks.core.feeding.bags.collection.base import BagsCollection
from arekit.contrib.networks.core.model import BaseTensorflowModel


class NetworksTrainingEngine(ExperimentEngine):

    def __init__(self, bags_collection_type, experiment,
                 load_model, config,
                 create_network_func,
                 prepare_model_root=True):
        assert(callable(create_network_func))
        assert(isinstance(config, DefaultNetworkConfig))
        assert(issubclass(bags_collection_type, BagsCollection))
        assert(isinstance(load_model, bool))
        super(NetworksTrainingEngine, self).__init__(experiment)

        self.__clear_model_root_before_experiment = prepare_model_root
        self.__config = config
        self.__create_network_func = create_network_func
        self.__bags_collection_type = bags_collection_type
        self.__load_model = load_model

    def __get_model_dir(self):
        return self._experiment.DataIO.ModelIO.get_model_dir()

    # region protected methods

    def _handle_iteration(self, it_index):
        """ Run single CV-index experiment.
        """
        assert(isinstance(it_index, int))

        # Perform data reading.
        handled_data = HandledData.create_empty()
        handled_data.perform_reading_and_initialization(experiment=self._experiment,
                                                        bags_collection_type=self.__bags_collection_type,
                                                        config=self.__config)

        # Setup callback
        callback = self._experiment.DataIO.Callback
        callback.on_experiment_iteration_begin()

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
        with callback:
            model.run_training(epochs_count=callback.Epochs)

        del network
        del model

        gc.collect()

    def _before_running(self):

        # Clear model root before training optionally
        if self.__clear_model_root_before_experiment:
            rm_dir_contents(dir_path=self.__get_model_dir(),
                            logger=self._logger)

        # Setup callback
        callback = self._experiment.DataIO.Callback
        callback.set_experiment(self._experiment)

        # Disable tensorflow logging
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # endregion