import logging
import os
import gc

import numpy as np

from arekit.common.experiment.engine.cv_based import ExperimentEngine
from arekit.common.experiment.engine.utils import rm_dir_contents
from arekit.contrib.networks.context.configurations.base.base import DefaultNetworkConfig
from arekit.contrib.networks.core.data_handling.data import HandledData
from arekit.contrib.networks.core.feeding.bags.collection.base import BagsCollection
from arekit.contrib.networks.core.model import BaseTensorflowModel
from arekit.contrib.networks.core.params import NeuralNetworkModelParams

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class NetworksTrainingEngine(ExperimentEngine):

    def __init__(self, bags_collection_type, experiment,
                 load_model, config,
                 create_network_func,
                 prepare_model_root=True,
                 seed=None):
        assert(callable(create_network_func))
        assert(isinstance(config, DefaultNetworkConfig))
        assert(issubclass(bags_collection_type, BagsCollection))
        assert(isinstance(load_model, bool))
        assert(isinstance(seed, int) or seed is None)

        super(NetworksTrainingEngine, self).__init__(experiment)

        self.__clear_model_root_before_experiment = prepare_model_root
        self.__config = config
        self.__create_network_func = create_network_func
        self.__bags_collection_type = bags_collection_type
        self.__load_model = load_model
        self.__seed = seed

    def __get_model_dir(self):
        return self._experiment.DataIO.ModelIO.get_model_dir()

    # region protected methods

    def _handle_iteration(self, it_index):
        """ Run single CV-index experiment.
        """
        assert(isinstance(it_index, int))

        # Perform data reading.
        handled_data = HandledData.create_empty()

        if not HandledData.check_files_existed(self._experiment):
            exp_folder_name = self._experiment.ExperimentIO.get_experiment_folder_name()
            raise Exception("Data has not been initialized/serialized: `{}`".format(exp_folder_name))

        # Reading embedding.
        embedding_filepath = self._experiment.ExperimentIO.get_loading_embedding_filepath()
        npz_embedding_data = np.load(embedding_filepath)
        self.__config.set_term_embedding(npz_embedding_data['arr_0'])
        logger.info("Embedding read [size={size}]: {filepath}".format(size=self.__config.TermEmbeddingMatrix.shape,
                                                                       filepath=embedding_filepath))

        # Reading vocabulary
        # TODO. This is suppose to be an external loader.
        # TODO. Provider, which returns dictionary.
        vocab_filepath = self._experiment.ExperimentIO.get_loading_vocab_filepath()
        npz_vocab_data = np.load(vocab_filepath)
        vocab = dict(npz_vocab_data['arr_0'])
        logger.info("Vocabulary read [size={size}]: {filepath}".format(size=len(vocab),
                                                                       filepath=vocab_filepath))

        # Performing samples reading process.
        handled_data.perform_reading_and_initialization(
            dtypes=self._experiment.DocumentOperations.DataFolding.iter_supported_data_types(),
            exp_io=self._experiment.ExperimentIO,
            labels_count=self._experiment.DataIO.LabelsCount,
            vocab=vocab,
            bags_collection_type=self.__bags_collection_type,
            config=self.__config)

        # Update parameters after iteration preparation has been completed.
        self.__config.reinit_config_dependent_parameters()

        # Setup callback.
        callback = self._experiment.DataIO.Callback
        callback.on_experiment_iteration_begin()

        # Initialize network and model.
        network = self.__create_network_func()
        model = BaseTensorflowModel(network=network,
                                    config=self.__config,
                                    handled_data=handled_data,
                                    bags_collection_type=self.__bags_collection_type,
                                    callback=callback,
                                    nn_io=self._experiment.DataIO.ModelIO)

        # Initialize model params instance.
        model_params = NeuralNetworkModelParams(epochs_count=callback.Epochs)

        # Run model
        with callback:
            model.run_training(model_params=model_params,
                               seed=self.__seed)

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

        # Notify other subscribers that initialization process has been completed.
        self.__config.init_initializers()

    def _after_running(self):
        self._experiment.DataIO.Callback.on_experiment_finished()

    # endregion