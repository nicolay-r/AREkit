import argparse
import os
from os.path import join

from arekit.common.data.input.providers.label.multiple import MultipleLabelProvider
from arekit.common.data.row_ids.multiple import MultipleIDProvider
from arekit.common.data.storages.base import BaseRowsStorage
from arekit.common.data.views.samples import BaseSampleStorageView
from arekit.common.experiment.data_type import DataType
from arekit.common.labels.scaler.base import BaseLabelScaler

from arekit.contrib.networks.core.ctx_inference import InferenceContext
from arekit.contrib.networks.core.feeding.bags.collection.single import SingleBagsCollection
from arekit.contrib.experiment_rusentrel.model_io.embedding import EmbeddingHelper
from arekit.contrib.networks.core.model import BaseTensorflowModel
from arekit.contrib.networks.core.model_io import NeuralNetworkModelIO
from arekit.contrib.networks.core.predict.provider import BasePredictProvider
from arekit.contrib.networks.core.predict.tsv_writer import TsvPredictWriter
from arekit.contrib.networks.enum_input_types import ModelInputType
from arekit.contrib.networks.enum_name_types import ModelNames
from arekit.contrib.networks.shapes import NetworkInputShapes

from arekit.processing.languages.ru.pos_service import PartOfSpeechTypesService

from examples.input import EXAMPLES
from examples.network.args import const
from examples.network.args.const import DATA_DIR
from examples.network.common import create_infer_experiment_name_provider
from examples.run_serialize import run_serializer
from examples.network.args.train import BagsPerMinibatchArg, ModelInputTypeArg, ModelNameTagArg
from examples.network.factory.networks import compose_network_and_network_config_funcs
from examples.network.train.common import Common
from examples.network.args.common import RusVectoresEmbeddingFilepathArg, \
    LabelsCountArg, TermsPerContextArg, \
    ModelNameArg


def run_infer(labels_scaler, bags_per_minibatch, model_name, model_input_type, model_name_tag):
    assert(isinstance(model_name, ModelNames))
    assert(isinstance(labels_scaler, BaseLabelScaler))
    assert(isinstance(model_input_type, ModelInputType))

    # Step 4. Deserialize data
    network_func, config_func = compose_network_and_network_config_funcs(
        model_name=model_name, model_input_type=model_input_type)

    network = network_func()
    config = config_func()
    exp_name_provider = create_infer_experiment_name_provider()
    exp_name = "{name}_{labels}l".format(name=exp_name_provider.provide(), labels=1)

    # Setup data filepaths.
    root = os.path.join(DATA_DIR, exp_name)

    samples_filepath = join(root, "sample-test-0.tsv.gz")
    embedding_filepath = join(root, "term_embedding-0.npz")
    result_filepath = join(root, "out.txt.gz")
    model_target_dir = ".model"
    vocab_filepath = join(root, const.VOCAB_DEFAULT)

    # Setup config parameters.
    embedding_matrix = EmbeddingHelper.load_embedding(embedding_filepath)
    config.set_term_embedding(embedding_matrix)
    config.set_pos_count(PartOfSpeechTypesService.get_mystem_pos_count())
    config.modify_classes_count(labels_scaler.LabelsCount)
    config.modify_bags_per_minibatch(bags_per_minibatch)
    config.set_class_weights([1, 1, 1])

    inference_ctx = InferenceContext.create_empty()
    inference_ctx.initialize(
        dtypes=[DataType.Test],
        bags_collection_type=SingleBagsCollection,
        create_samples_view_func=lambda data_type: BaseSampleStorageView(
            storage=BaseRowsStorage.from_tsv(samples_filepath),
            row_ids_provider=MultipleIDProvider()),
        has_model_predefined_state=True,
        vocab=EmbeddingHelper.load_vocab(vocab_filepath),
        labels_count=config.ClassesCount,
        input_shapes=NetworkInputShapes(iter_pairs=[
            (NetworkInputShapes.FRAMES_PER_CONTEXT, config.FramesPerContext),
            (NetworkInputShapes.TERMS_PER_CONTEXT, config.TermsPerContext),
            (NetworkInputShapes.SYNONYMS_PER_CONTEXT, config.SynonymsPerContext),
        ]),
        bag_size=config.BagSize)

    # Step 5. Model preparation.
    model = BaseTensorflowModel(
        nn_io=NeuralNetworkModelIO(target_dir=model_target_dir,
                                   full_model_name=model_name.value,
                                   model_name_tag=model_name_tag),
        network=network,
        config=config,
        inference_ctx=inference_ctx,
        bags_collection_type=SingleBagsCollection,      # Используем на вход 1 пример.
    )

    model.predict(do_compile=True)

    # Step 6. Gather annotated contexts onto document level.
    labeled_samples = model.get_labeled_samples_collection(data_type=DataType.Test)

    predict_provider = BasePredictProvider()

    # TODO. For now it is limited to tsv.
    with TsvPredictWriter(filepath=result_filepath) as out:

        title, contents_it = predict_provider.provide(
            sample_id_with_uint_labels_iter=labeled_samples.iter_non_duplicated_labeled_sample_row_ids(),
            labels_scaler=labels_scaler)

        out.write(title=title,
                  contents_it=contents_it)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Inference")

    # Providing arguments.
    RusVectoresEmbeddingFilepathArg.add_argument(parser)
    BagsPerMinibatchArg.add_argument(parser)
    LabelsCountArg.add_argument(parser)
    ModelNameArg.add_argument(parser)
    ModelNameTagArg.add_argument(parser)
    ModelInputTypeArg.add_argument(parser)
    TermsPerContextArg.add_argument(parser)

    # Parsing arguments.
    args = parser.parse_args()

    # Reading provided arguments.
    rusvectores_embedding_path = RusVectoresEmbeddingFilepathArg.read_argument(args)
    bags_per_minibatch = BagsPerMinibatchArg.read_argument(args)
    labels_count = LabelsCountArg.read_argument(args)
    model_name = ModelNameArg.read_argument(args)
    model_name_tag = ModelNameTagArg.read_argument(args)
    model_input_type = ModelInputTypeArg.read_argument(args)
    terms_per_context = TermsPerContextArg.read_argument(args)

    # Implement extra structures.
    labels_scaler = Common.create_labels_scaler(labels_count)
    label_provider = MultipleLabelProvider(label_scaler=labels_scaler)

    # Parsing arguments.
    args = parser.parse_args()

    # Execute pipeline elements.
    run_serializer(sentences_text_list=EXAMPLES["simple"],
                   embedding_path=rusvectores_embedding_path,
                   terms_per_context=terms_per_context)

    run_infer(labels_scaler=labels_scaler,
              bags_per_minibatch=bags_per_minibatch,
              model_name=model_name,
              model_input_type=model_input_type,
              model_name_tag=model_name_tag)
