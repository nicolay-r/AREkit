import logging
import os
import gc

from arekit.common.experiment.engine import ExperimentEngine
from arekit.contrib.networks.context.configurations.base.base import DefaultNetworkConfig
from arekit.contrib.networks.core.ctx_inference import InferenceContext
from arekit.contrib.networks.core.feeding.bags.collection.base import BagsCollection
from arekit.contrib.networks.core.model import BaseTensorflowModel
from arekit.contrib.networks.core.model_ctx import TensorflowModelContext
from arekit.contrib.networks.core.params import NeuralNetworkModelParams
from arekit.contrib.networks.core.pipeline_fit import MinibatchFittingPipelineItem
from arekit.contrib.networks.core.pipeline_keep_hidden import MinibatchHiddenFetcherPipelineItem
from arekit.contrib.networks.core.pipeline_predict import EpochLabelsPredictorPipelineItem
from arekit.contrib.networks.core.pipeline_predict_labeling import EpochLabelsCollectorPipelineItem
from arekit.contrib.networks.shapes import NetworkInputShapes
from arekit.contrib.networks.utils import rm_dir_contents

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class NetworksTrainingEngine(ExperimentEngine):

    def __init__(self, bags_collection_type, experiment,
                 load_model, config,
                 create_network_func,
                 training_epochs,
                 prepare_model_root=True,
                 seed=None):
        assert(callable(create_network_func))
        assert(isinstance(config, DefaultNetworkConfig))
        assert(issubclass(bags_collection_type, BagsCollection))
        assert(isinstance(load_model, bool))
        assert(isinstance(seed, int) or seed is None)
        assert(isinstance(training_epochs, int))

        super(NetworksTrainingEngine, self).__init__(experiment)

        self.__clear_model_root_before_experiment = prepare_model_root
        self.__config = config
        self.__create_network_func = create_network_func
        self.__bags_collection_type = bags_collection_type
        self.__load_model = load_model
        self.__training_epochs = training_epochs
        self.__seed = seed

    def __get_model_dir(self):
        return self._experiment.DataIO.ModelIO.get_model_dir()

    # region protected methods

    def _handle_iteration(self, it_index):
        """ Run single CV-index experiment.
        """
        assert(isinstance(it_index, int))

        targets_existed = self._experiment.ExperimentIO.check_targets_existed(
            data_types_iter=self._experiment.DocumentOperations.DataFolding.iter_supported_data_types())

        if not targets_existed:
            raise Exception("Data has not been initialized/serialized!")

        # Reading embedding.
        embedding_data = self._experiment.ExperimentIO.load_embedding()
        self.__config.set_term_embedding(embedding_data)

        # Performing samples reading process.
        inference_ctx = InferenceContext.create_empty()
        inference_ctx.initialize(
            dtypes=self._experiment.DocumentOperations.DataFolding.iter_supported_data_types(),
            create_samples_view_func=lambda data_type: self._experiment.ExperimentIO.create_samples_view(data_type),
            has_model_predefined_state=self._experiment.ExperimentIO.has_model_predefined_state(),
            labels_count=self._experiment.DataIO.LabelsCount,
            vocab=self._experiment.ExperimentIO.load_vocab(),
            bags_collection_type=self.__bags_collection_type,
            input_shapes=NetworkInputShapes(iter_pairs=[
                (NetworkInputShapes.FRAMES_PER_CONTEXT, self.__config.FramesPerContext),
                (NetworkInputShapes.TERMS_PER_CONTEXT, self.__config.TermsPerContext),
                (NetworkInputShapes.SYNONYMS_PER_CONTEXT, self.__config.SynonymsPerContext),
            ]),
            bag_size=self.__config.BagSize)

        if inference_ctx.HasNormalizedWeights:
            weights = inference_ctx.calc_normalized_weigts(labels_count=self._experiment.DataIO.LabelsCount)
            self.__config.set_class_weights(weights)

        # Update parameters after iteration preparation has been completed.
        self.__config.reinit_config_dependent_parameters()

        # Setup callback.
        callback = self._experiment.DataIO.Callback
        callback.on_experiment_iteration_begin()

        # Initialize network and model.
        network = self.__create_network_func()
        model = BaseTensorflowModel(
            context=TensorflowModelContext(
                network=network,
                config=self.__config,
                inference_ctx=inference_ctx,
                bags_collection_type=self.__bags_collection_type,
                nn_io=self._experiment.DataIO.ModelIO),
            callback=callback,
            predict_pipeline=[
                EpochLabelsPredictorPipelineItem(),
                EpochLabelsCollectorPipelineItem(),
                MinibatchHiddenFetcherPipelineItem()
            ],
            fit_pipeline=[MinibatchFittingPipelineItem()])

        # Initialize model params instance.
        model_params = NeuralNetworkModelParams(epochs_count=self.__training_epochs)

        # Run model
        with callback:
            model.fit(model_params=model_params,
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
