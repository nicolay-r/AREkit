from os.path import join

from arekit.common.data.input.providers.label.multiple import MultipleLabelProvider
from arekit.common.data.row_ids.multiple import MultipleIDProvider
from arekit.common.data.storages.base import BaseRowsStorage
from arekit.common.data.views.samples import BaseSampleStorageView
from arekit.common.experiment.data_type import DataType
from arekit.common.labels.scaler.base import BaseLabelScaler

from arekit.contrib.experiment_rusentrel.labels.scalers.three import ThreeLabelScaler
from arekit.contrib.networks.context.architectures.pcnn import PiecewiseCNN
from arekit.contrib.networks.context.configurations.cnn import CNNConfig
from arekit.contrib.networks.core.ctx_inference import InferenceContext
from arekit.contrib.networks.core.feeding.bags.collection.single import SingleBagsCollection
from arekit.contrib.networks.core.input.helper_embedding import EmbeddingHelper
from arekit.contrib.networks.core.model import BaseTensorflowModel
from arekit.contrib.networks.core.model_io import NeuralNetworkModelIO
from arekit.contrib.networks.core.predict.provider import BasePredictProvider
from arekit.contrib.networks.core.predict.tsv_writer import TsvPredictWriter

from arekit.contrib.networks.shapes import NetworkInputShapes
from arekit.processing.languages.ru.pos_service import PartOfSpeechTypesService

from examples.input import EXAMPLES
from examples.run_serialize import pipeline_serialize


def pipeline_infer(labels_scaler):
    assert(isinstance(labels_scaler, BaseLabelScaler))

    # Step 4. Deserialize data
    network = PiecewiseCNN()
    config = CNNConfig()

    # setup data filepaths.
    root = "data/test-test_1l/"
    samples_filepath = join(root, "sample-test-0.tsv.gz")
    embedding_filepath = join(root, "term_embedding-0.npz")
    vocab_filepath = join(root, "vocab-0.txt.npz")
    result_filepath = join(root, "out.txt.gz")

    # setup config parameters.
    embedding = EmbeddingHelper.load_embedding(embedding_filepath)
    config.set_term_embedding(embedding)
    config.set_pos_count(PartOfSpeechTypesService.get_mystem_pos_count())
    config.modify_classes_count(labels_scaler.LabelsCount)
    config.modify_bags_per_minibatch(3)
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
        nn_io=NeuralNetworkModelIO(
            target_dir=".model",
            full_model_name="PCNN",
            model_name_tag="_"),
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

    text = EXAMPLES["simple"]
    labels_scaler = ThreeLabelScaler()
    label_provider = MultipleLabelProvider(label_scaler=labels_scaler)

    pipeline_serialize(sentences_text_list=text)
    pipeline_infer(labels_scaler)
