import argparse
import os
from os.path import join

from arekit.common.data.row_ids.multiple import MultipleIDProvider
from arekit.common.data.storages.base import BaseRowsStorage
from arekit.common.data.views.samples import BaseSampleStorageView
from arekit.common.experiment.data_type import DataType

from arekit.contrib.networks.core.ctx_inference import InferenceContext
from arekit.contrib.networks.core.model import BaseTensorflowModel
from arekit.contrib.networks.core.model_ctx import TensorflowModelContext
from arekit.contrib.networks.core.network_callback import NetworkCallback
from arekit.contrib.networks.core.pipeline_fit import MinibatchFittingPipelineItem
from arekit.contrib.networks.core.pipeline_keep_hidden import MinibatchHiddenFetcherPipelineItem
from arekit.contrib.networks.core.pipeline_predict import EpochLabelsPredictorPipelineItem
from arekit.contrib.networks.core.predict.provider import BasePredictProvider
from arekit.contrib.networks.core.predict.tsv_writer import TsvPredictWriter
from arekit.contrib.networks.factory import create_network_and_network_config_funcs
from arekit.contrib.networks.shapes import NetworkInputShapes

from arekit.processing.languages.ru.pos_service import PartOfSpeechTypesService

from examples.input import EXAMPLES
from examples.network.args.const import NEURAL_NETWORKS_TARGET_DIR, BAG_SIZE
from examples.network.args.serialize import EntityFormatterTypesArg
from examples.network.args.train import BagsPerMinibatchArg, ModelInputTypeArg, ModelNameTagArg
from examples.network.common import create_bags_collection_type, create_network_model_io
from examples.network.args.common import RusVectoresEmbeddingFilepathArg, LabelsCountArg, TermsPerContextArg, \
    ModelNameArg, ModelLoadDirArg, VocabFilepathArg, StemmerArg, InputTextArg, PredictOutputFilepathArg, \
    EmbeddingMatrixFilepathArg
from examples.network.infer.exp_io import InferIOUtils
from examples.run_text_serialize import run_serializer
from examples.rusentrel.common import Common


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Text inference example")

    # Providing arguments.
    InputTextArg.add_argument(parser, default=EXAMPLES["simple"][1])
    RusVectoresEmbeddingFilepathArg.add_argument(parser)
    BagsPerMinibatchArg.add_argument(parser)
    LabelsCountArg.add_argument(parser)
    ModelNameArg.add_argument(parser)
    ModelNameTagArg.add_argument(parser)
    ModelInputTypeArg.add_argument(parser)
    TermsPerContextArg.add_argument(parser)
    EntityFormatterTypesArg.add_argument(parser)
    VocabFilepathArg.add_argument(parser, default=None)
    EmbeddingMatrixFilepathArg.add_argument(parser, default=None)
    ModelLoadDirArg.add_argument(parser, default=NEURAL_NETWORKS_TARGET_DIR)
    StemmerArg.add_argument(parser)
    PredictOutputFilepathArg.add_argument(parser, default=None)

    # Parsing arguments.
    args = parser.parse_args()

    # Reading provided arguments.
    text = InputTextArg.read_argument(args)
    rusvectores_embedding_path = RusVectoresEmbeddingFilepathArg.read_argument(args)
    bags_per_minibatch = BagsPerMinibatchArg.read_argument(args)
    labels_count = LabelsCountArg.read_argument(args)
    model_name = ModelNameArg.read_argument(args)
    model_name_tag = ModelNameTagArg.read_argument(args)
    model_input_type = ModelInputTypeArg.read_argument(args)
    terms_per_context = TermsPerContextArg.read_argument(args)
    entity_fmt_type = EntityFormatterTypesArg.read_argument(args)
    stemmer = StemmerArg.read_argument(args)
    model_load_dir = ModelLoadDirArg.read_argument(args)
    result_filepath = PredictOutputFilepathArg.read_argument(args)
    vocab_filepath = VocabFilepathArg.read_argument(args)
    embedding_matrix_filepath = EmbeddingMatrixFilepathArg.read_argument(args)

    # Implement extra structures.
    labels_scaler = Common.create_labels_scaler(labels_count)

    # Initialize bags collection type.
    bags_collection_type = create_bags_collection_type(model_input_type=model_input_type)

    # Parsing arguments.
    args = parser.parse_args()

    # Execute pipeline elements.
    serialized_exp_io = run_serializer(sentences_text_list=[text],
                                       embedding_path=rusvectores_embedding_path,
                                       terms_per_context=terms_per_context,
                                       entity_fmt_type=entity_fmt_type,
                                       stemmer=stemmer)

    assert(isinstance(serialized_exp_io, InferIOUtils))

    # Create network an configuration.
    network_func, config_func = create_network_and_network_config_funcs(
        model_name=model_name, model_input_type=model_input_type)

    network = network_func()
    config = config_func()

    # Declaring result filepath.
    if result_filepath is None:
        root = os.path.join(serialized_exp_io._get_experiment_sources_dir(),
                            serialized_exp_io.get_experiment_folder_name())
        result_filepath = join(root, "out.tsv.gz")

    full_model_name = Common.create_full_model_name(
        model_name=model_name,
        input_type=model_input_type)

    nn_io = create_network_model_io(full_model_name=full_model_name,
                                    source_dir=model_load_dir,
                                    embedding_filepath=embedding_matrix_filepath,
                                    target_dir=model_load_dir,
                                    vocab_filepath=vocab_filepath,
                                    model_name_tag=model_name_tag)

    samples_filepath = serialized_exp_io.create_samples_writer_target(DataType.Test)

    # Setup config parameters.
    config.set_term_embedding(serialized_exp_io.load_embedding())
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
        vocab=serialized_exp_io.load_vocab(),
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
        callback=NetworkCallback(),
        predict_pipeline=[
            EpochLabelsPredictorPipelineItem(),
            MinibatchHiddenFetcherPipelineItem()
        ],
        fit_pipeline=[MinibatchFittingPipelineItem()])

    model.predict(do_compile=True)

    # Gather annotated contexts onto document level.
    item = model.from_predicted(EpochLabelsPredictorPipelineItem)
    labeled_samples = item.LabeledSamples

    predict_provider = BasePredictProvider()

    # Saving Results.
    # TODO. For now it is limited to tsv.
    with TsvPredictWriter(filepath=result_filepath) as out:

        title, contents_it = predict_provider.provide(
            sample_id_with_uint_labels_iter=labeled_samples.iter_non_duplicated_labeled_sample_row_ids(),
            labels_scaler=labels_scaler)

        out.write(title=title,
                  contents_it=contents_it)
