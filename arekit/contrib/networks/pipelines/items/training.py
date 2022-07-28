import gc
import logging
import os

from arekit.common.experiment.data_type import DataType
from arekit.common.folding.base import BaseDataFolding
from arekit.common.pipeline.context import PipelineContext
from arekit.common.pipeline.items.base import BasePipelineItem
from arekit.contrib.networks.context.configurations.base.base import DefaultNetworkConfig
from arekit.contrib.networks.core.ctx_inference import InferenceContext
from arekit.contrib.networks.core.feeding.bags.collection.base import BagsCollection
from arekit.contrib.networks.core.model import BaseTensorflowModel
from arekit.contrib.networks.core.model_ctx import TensorflowModelContext
from arekit.contrib.networks.core.params import NeuralNetworkModelParams
from arekit.contrib.networks.core.pipeline.item_fit import MinibatchFittingPipelineItem
from arekit.contrib.networks.core.pipeline.item_keep_hidden import MinibatchHiddenFetcherPipelineItem
from arekit.contrib.networks.core.pipeline.item_predict import EpochLabelsPredictorPipelineItem
from arekit.contrib.networks.core.pipeline.item_predict_labeling import EpochLabelsCollectorPipelineItem
from arekit.contrib.networks.shapes import NetworkInputShapes
from arekit.contrib.networks.utils import rm_dir_contents

from arekit.contrib.utils.model_io.tf_networks import DefaultNetworkIOUtils
from arekit.contrib.utils.utils_folding import folding_iter_states


class NetworksTrainingPipelineItem(BasePipelineItem):

    def __init__(self, bags_collection_type, model_io, exp_io,
                 load_model, config, create_network_func, training_epochs,
                 labels_count, network_callbacks, prepare_model_root=True, seed=None):
        assert(callable(create_network_func))
        assert(isinstance(exp_io, DefaultNetworkIOUtils))
        assert(isinstance(config, DefaultNetworkConfig))
        assert(issubclass(bags_collection_type, BagsCollection))
        assert(isinstance(load_model, bool))
        assert(isinstance(seed, int) or seed is None)
        assert(isinstance(training_epochs, int))
        assert(isinstance(network_callbacks, list))
        assert(isinstance(labels_count, int))

        super(NetworksTrainingPipelineItem, self).__init__()

        self.__logger = self.__create_logger()
        self.__exp_io = exp_io
        self.__clear_model_root_before_experiment = prepare_model_root
        self.__config = config
        self.__create_network_func = create_network_func
        self.__bags_collection_type = bags_collection_type
        self.__network_callbacks = network_callbacks
        self.__load_model = load_model
        self.__training_epochs = training_epochs
        self.__labels_count = labels_count
        self.__model_io = model_io
        self.__seed = seed

    def __get_model_dir(self):
        return self.__model_io.get_model_dir()

    @staticmethod
    def __create_logger():
        stream_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s %(levelname)8s %(name)s | %(message)s')
        stream_handler.setFormatter(formatter)
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        logger.addHandler(stream_handler)
        return logger

    def __prepare_model(self):
        # Clear model root before training optionally
        if self.__clear_model_root_before_experiment:
            rm_dir_contents(dir_path=self.__get_model_dir(),
                            logger=self.__logger)

        # Disable tensorflow logging
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        # Notify other subscribers that initialization process has been completed.
        self.__config.init_initializers()

    def __handle_iteration(self, data_folding, data_type):
        assert(isinstance(data_folding, BaseDataFolding))
        assert(isinstance(data_type, DataType))

        targets_existed = self.__exp_io.check_targets_existed(
            data_types_iter=data_folding.iter_supported_data_types(),
            data_folding=data_folding)

        if not targets_existed:
            raise Exception("Data has not been initialized/serialized!")

        # Reading embedding.
        embedding_data = self.__exp_io.load_embedding(data_folding)
        self.__config.set_term_embedding(embedding_data)

        # Performing samples reading process.
        inference_ctx = InferenceContext.create_empty()
        inference_ctx.initialize(
            dtypes=data_folding.iter_supported_data_types(),
            create_samples_view_func=lambda data_type: self.__exp_io.create_samples_view(
                data_type=data_type, data_folding=data_folding),
            has_model_predefined_state=self.__exp_io.has_model_predefined_state(),
            labels_count=self.__labels_count,
            vocab=self.__exp_io.load_vocab(data_folding),
            bags_collection_type=self.__bags_collection_type,
            input_shapes=NetworkInputShapes(iter_pairs=[
                (NetworkInputShapes.FRAMES_PER_CONTEXT, self.__config.FramesPerContext),
                (NetworkInputShapes.TERMS_PER_CONTEXT, self.__config.TermsPerContext),
                (NetworkInputShapes.SYNONYMS_PER_CONTEXT, self.__config.SynonymsPerContext),
            ]),
            bag_size=self.__config.BagSize)

        if inference_ctx.HasNormalizedWeights:
            weights = inference_ctx.calc_normalized_weigts(labels_count=self.__labels_count)
            self.__config.set_class_weights(weights)

        # Update parameters after iteration preparation has been completed.
        self.__config.reinit_config_dependent_parameters()

        # Initialize network and model.
        network = self.__create_network_func()
        model = BaseTensorflowModel(
            context=TensorflowModelContext(
                network=network,
                config=self.__config,
                inference_ctx=inference_ctx,
                bags_collection_type=self.__bags_collection_type,
                nn_io=self.__model_io),
            callbacks=self.__network_callbacks,
            predict_pipeline=[
                EpochLabelsPredictorPipelineItem(),
                EpochLabelsCollectorPipelineItem(),
                MinibatchHiddenFetcherPipelineItem()
            ],
            fit_pipeline=[
                MinibatchFittingPipelineItem(),
                MinibatchHiddenFetcherPipelineItem()
            ])

        # Initialize model params instance.
        model_params = NeuralNetworkModelParams(epochs_count=self.__training_epochs)

        model.fit(model_params=model_params, seed=self.__seed, data_type=data_type)

        del network
        del model

        gc.collect()

    def apply_core(self, input_data, pipeline_ctx):
        assert(isinstance(pipeline_ctx, PipelineContext))
        assert("data_folding" in pipeline_ctx)

        # Prepare all the required data.
        self.__prepare_model()
        data_folding = pipeline_ctx.provide("data_folding")
        data_type = pipeline_ctx.provide_or_none("data_type")
        for _ in folding_iter_states(data_folding):
            self.__handle_iteration(data_folding=data_folding,
                                    data_type=data_type if data_type is not None else DataType.Train)
