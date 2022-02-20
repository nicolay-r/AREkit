from arekit.common.data.row_ids.multiple import MultipleIDProvider
from arekit.common.data.storages.base import BaseRowsStorage
from arekit.common.data.views.samples import BaseSampleStorageView
from arekit.common.experiment.data_type import DataType
from arekit.contrib.networks.core.callback.stat import TrainingStatProviderCallback
from arekit.contrib.networks.core.callback.train_limiter import TrainingLimiterCallback
from arekit.contrib.networks.core.callback.writer import PredictResultWriterCallback
from arekit.contrib.networks.core.ctx_inference import InferenceContext
from arekit.contrib.networks.core.model import BaseTensorflowModel
from arekit.contrib.networks.core.model_ctx import TensorflowModelContext
from arekit.contrib.networks.core.pipeline.item_fit import MinibatchFittingPipelineItem
from arekit.contrib.networks.core.pipeline.item_keep_hidden import MinibatchHiddenFetcherPipelineItem
from arekit.contrib.networks.core.pipeline.item_predict import EpochLabelsPredictorPipelineItem
from arekit.contrib.networks.core.pipeline.item_predict_labeling import EpochLabelsCollectorPipelineItem
from arekit.contrib.networks.factory import create_network_and_network_config_funcs
from arekit.contrib.networks.shapes import NetworkInputShapes
from arekit.processing.languages.ru.pos_service import PartOfSpeechTypesService
from examples.network.args.const import BAG_SIZE
from examples.network.infer.exp_io import InferIOUtils


# TODO. #285 reorganize in a form of a pipeline item.
def run_network_inference_pipeline(serialized_exp_io, model_name, bags_collection_type,
                                   model_input_type, bags_per_minibatch, nn_io, labels_scaler,
                                   predict_writer):
    """ NeuralNetwork-based inference example.
    """
    assert (isinstance(serialized_exp_io, InferIOUtils))

    # Create network an configuration.
    network_func, config_func = create_network_and_network_config_funcs(
        model_name=model_name, model_input_type=model_input_type)

    network = network_func()
    config = config_func()

    samples_filepath = serialized_exp_io.create_samples_writer_target(DataType.Test)
    embedding = serialized_exp_io.load_embedding()
    vocab = serialized_exp_io.load_vocab()

    # Setup config parameters.
    config.set_term_embedding(embedding)
    config.set_pos_count(PartOfSpeechTypesService.get_mystem_pos_count())
    config.modify_classes_count(labels_scaler.LabelsCount)
    config.modify_bag_size(BAG_SIZE)
    config.modify_bags_per_minibatch(bags_per_minibatch)
    config.set_class_weights([1, 1, 1])

    inference_ctx = InferenceContext.create_empty()
    inference_ctx.initialize(
        dtypes=[DataType.Test],
        bags_collection_type=bags_collection_type,
        create_samples_view_func=lambda data_type: BaseSampleStorageView(
            storage=BaseRowsStorage.from_tsv(samples_filepath),
            row_ids_provider=MultipleIDProvider()),
        has_model_predefined_state=True,
        vocab=vocab,
        labels_count=config.ClassesCount,
        input_shapes=NetworkInputShapes(iter_pairs=[
            (NetworkInputShapes.FRAMES_PER_CONTEXT, config.FramesPerContext),
            (NetworkInputShapes.TERMS_PER_CONTEXT, config.TermsPerContext),
            (NetworkInputShapes.SYNONYMS_PER_CONTEXT, config.SynonymsPerContext),
        ]),
        bag_size=config.BagSize)

    # Model preparation.
    model = BaseTensorflowModel(
        context=TensorflowModelContext(
            nn_io=nn_io,
            network=network,
            config=config,
            inference_ctx=inference_ctx,
            bags_collection_type=bags_collection_type),
        callbacks=[
            TrainingLimiterCallback(train_acc_limit=0.99),
            TrainingStatProviderCallback(),
            PredictResultWriterCallback(labels_scaler=labels_scaler, writer=predict_writer)
        ],
        predict_pipeline=[
            EpochLabelsPredictorPipelineItem(),
            EpochLabelsCollectorPipelineItem(),
            MinibatchHiddenFetcherPipelineItem()
        ],
        fit_pipeline=[MinibatchFittingPipelineItem()])

    model.predict(do_compile=True)
