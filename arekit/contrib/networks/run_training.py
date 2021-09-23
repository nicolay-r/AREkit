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
from arekit.contrib.networks.shapes import NetworkInputShapes

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

        targets_existed = self._experiment.ExperimentIO.check_targets_existed(
            data_types_iter=self._experiment.DocumentOperations.DataFolding.iter_supported_data_types())

        if not targets_existed:
            exp_folder_name = self._experiment.ExperimentIO.get_experiment_folder_name()
            raise Exception("Data has not been initialized/serialized: `{}`".format(exp_folder_name))

        # Reading embedding.
        # TODO: 200. Organize embedding storage.
        embedding_filepath = self._experiment.ExperimentIO.get_term_embedding_source()
        npz_embedding_data = np.load(embedding_filepath)
        self.__config.set_term_embedding(npz_embedding_data['arr_0'])
        logger.info("Embedding read [size={size}]: {filepath}".format(size=self.__config.TermEmbeddingMatrix.shape,
                                                                      filepath=embedding_filepath))

        # Reading vocabulary
        # TODO: 200. Organize vocab storage.
        vocab_source = self._experiment.ExperimentIO.get_vocab_source()
        npz_vocab_data = np.load(vocab_source)
        vocab = dict(npz_vocab_data['arr_0'])
        logger.info("Vocabulary read [size={size}]: {filepath}".format(size=len(vocab),
                                                                       filepath=vocab_source))

        # Performing samples reading process.
        handled_data.initialize(
            dtypes=self._experiment.DocumentOperations.DataFolding.iter_supported_data_types(),
            create_samples_reader_func=lambda data_type: self._experiment.ExperimentIO.create_samples_reader(data_type),
            has_model_predefined_state=self._experiment.ExperimentIO.has_model_predefined_state,
            exp_io=self._experiment.ExperimentIO,
            labels_count=self._experiment.DataIO.LabelsCount,
            vocab=vocab,
            bags_collection_type=self.__bags_collection_type,
            input_shapes=NetworkInputShapes(iter_pairs=[
                (NetworkInputShapes.FRAMES_PER_CONTEXT, self.__config.FramesPerContext),
                (NetworkInputShapes.TERMS_PER_CONTEXT, self.__config.TermsPerContext),
                (NetworkInputShapes.SYNONYMS_PER_CONTEXT, self.__config.SynonymsPerContext),
            ]),
            bag_size=self.__config.BagSize)

        if handled_data.HasNormalizedWeights:
            weights = handled_data.calc_normalized_weigts(labels_count=self._experiment.DataIO.LabelsCount)
            self.__config.set_class_weights(weights)

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