import argparse
import os
from os.path import join

from arekit.contrib.networks.core.predict.tsv_writer import TsvPredictWriter
from examples.input import EXAMPLES
from examples.network.args.const import NEURAL_NETWORKS_TARGET_DIR
from examples.network.args.serialize import EntityFormatterTypesArg
from examples.network.args.train import BagsPerMinibatchArg, ModelInputTypeArg, ModelNameTagArg
from examples.network.common import create_bags_collection_type, create_network_model_io
from examples.network.args.common import RusVectoresEmbeddingFilepathArg, LabelsCountArg, TermsPerContextArg, \
    ModelNameArg, ModelLoadDirArg, VocabFilepathArg, StemmerArg, InputTextArg, PredictOutputFilepathArg, \
    EmbeddingMatrixFilepathArg, EntitiesParserArg
from examples.pipelines.inference import run_network_inference_pipeline

from examples.pipelines.serialize import run_data_serialization_pipeline
from examples.rusentrel.common import Common


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Text inference example")

    # Providing arguments.
    InputTextArg.add_argument(parser, default=EXAMPLES["no_entities"])
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
    EntitiesParserArg.add_argument(parser, default="bert-ontonotes")
    StemmerArg.add_argument(parser)
    PredictOutputFilepathArg.add_argument(parser, default=None)

    # Parsing arguments.
    args = parser.parse_args()

    # Reading provided arguments.
    sentences = InputTextArg.read_argument(args)
    rusvectores_embedding_path = RusVectoresEmbeddingFilepathArg.read_argument(args)
    bags_per_minibatch = BagsPerMinibatchArg.read_argument(args)
    labels_count = LabelsCountArg.read_argument(args)
    model_name = ModelNameArg.read_argument(args)
    model_name_tag = ModelNameTagArg.read_argument(args)
    model_input_type = ModelInputTypeArg.read_argument(args)
    entities_parser = EntitiesParserArg.read_argument(args)
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

    #############################
    # Execute pipeline element.
    #############################
    serialized_exp_io = run_data_serialization_pipeline(
        sentences=sentences,
        embedding_path=rusvectores_embedding_path,
        terms_per_context=terms_per_context,
        entity_fmt_type=entity_fmt_type,
        entities_parser=entities_parser,
        stemmer=stemmer)

    #############################
    # Execute pipeline element.
    #############################
    full_model_name = Common.create_full_model_name(model_name=model_name,
                                                    input_type=model_input_type)

    nn_io = create_network_model_io(full_model_name=full_model_name,
                                    source_dir=model_load_dir,
                                    embedding_filepath=embedding_matrix_filepath,
                                    target_dir=model_load_dir,
                                    vocab_filepath=vocab_filepath,
                                    model_name_tag=model_name_tag)

    # Setup predicted result writer.
    if result_filepath is None:
        root = os.path.join(serialized_exp_io._get_experiment_sources_dir(),
                            serialized_exp_io.get_experiment_folder_name())
        result_filepath = join(root, "out.tsv.gz")
    writer = TsvPredictWriter(result_filepath)

    run_network_inference_pipeline(serialized_exp_io=serialized_exp_io,
                                   model_name=model_name,
                                   bags_collection_type=bags_collection_type,
                                   predict_writer=writer,
                                   bags_per_minibatch=bags_per_minibatch,
                                   model_input_type=model_input_type,
                                   nn_io=nn_io,
                                   labels_scaler=labels_scaler)
