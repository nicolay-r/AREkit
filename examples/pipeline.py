from arekit.common.data.input.providers.label.multiple import MultipleLabelProvider
from arekit.common.data.input.providers.text.single import BaseSingleTextProvider
from arekit.common.data.row_ids.multiple import MultipleIDProvider
from arekit.common.data.storages.base import BaseRowsStorage
from arekit.common.data.views.samples import BaseSampleStorageView
from arekit.common.experiment.annot.single_label import DefaultSingleLabelAnnotationAlgorithm
from arekit.common.experiment.data_type import DataType
from arekit.common.frame_variants.collection import FrameVariantsCollection
from arekit.common.labels.base import NoLabel
from arekit.common.news.base import News
from arekit.common.text.options import TextParseOptions

from arekit.contrib.experiment_rusentrel.common import entity_to_group_func
from arekit.contrib.experiment_rusentrel.entities.str_simple_fmt import StringEntitiesSimpleFormatter
from arekit.contrib.experiment_rusentrel.labels.scalers.three import ThreeLabelScaler
from arekit.contrib.experiment_rusentrel.synonyms.provider import RuSentRelSynonymsCollectionProvider
from arekit.contrib.networks.context.architectures.pcnn import PiecewiseCNN
from arekit.contrib.networks.context.configurations.cnn import CNNConfig
from arekit.contrib.networks.core.ctx_inference import InferenceContext
from arekit.contrib.networks.core.feeding.bags.collection.single import SingleBagsCollection
from arekit.contrib.networks.core.input.formatters.pos_mapper import PosTermsMapper
from arekit.contrib.networks.core.input.helper import NetworkInputHelper
from arekit.contrib.networks.core.input.helper_embedding import EmbeddingHelper
from arekit.contrib.networks.core.input.providers.sample import NetworkSampleRowProvider
from arekit.contrib.networks.core.input.terms_mapping import StringWithEmbeddingNetworkTermMapping
from arekit.contrib.networks.core.model import BaseTensorflowModel
from arekit.contrib.networks.core.model_io import NeuralNetworkModelIO
from arekit.contrib.networks.core.predict.tsv_provider import TsvPredictProvider
from arekit.contrib.networks.shapes import NetworkInputShapes
from arekit.contrib.source.rusentiframes.collection import RuSentiFramesCollection
from arekit.contrib.source.rusentiframes.types import RuSentiFramesVersions
from arekit.contrib.source.rusentrel.io_utils import RuSentRelVersions

from arekit.processing.lemmatization.mystem import MystemWrapper
from arekit.processing.text.parser import DefaultTextParser

from examples.input import EXAMPLES
from examples.network.embedding import RusvectoresEmbedding


def extract(text):

    ########################
    # Step 1. Parse text.
    ########################

    sentences = text  # TODO. split text onto sentences.
    stemmer = MystemWrapper()

    news = News(doc_id=0,
                sentences=sentences)

    parse_options = TextParseOptions(
        parse_entities=False,
        frame_variants_collection=FrameVariantsCollection(),
        stemmer=stemmer)

    text_parser = DefaultTextParser(parse_options)

    parsed_news = text_parser.parse_news(news=news)

    ########################
    # Step 2. Annotate text.
    ########################

    synonyms = RuSentRelSynonymsCollectionProvider.load_collection(
        stemmer=stemmer,
        version=RuSentRelVersions.V11)

    labels_scaler = ThreeLabelScaler()

    annot_algo = DefaultSingleLabelAnnotationAlgorithm(
        dist_in_terms_bound=None,
        label_instance=NoLabel())

    opins_for_extraction = annot_algo.iter_opinions(
        parsed_news=parsed_news,
        entities_collection=None)   # TODO. Create custom entity collections.

    sample_row_provider = NetworkSampleRowProvider(
        label_provider=MultipleLabelProvider(label_scaler=labels_scaler),
        text_provider=BaseSingleTextProvider(
            text_terms_mapper=StringWithEmbeddingNetworkTermMapping(
                entity_to_group_func=lambda entity: entity_to_group_func(entity, synonyms),
                predefined_embedding=RusvectoresEmbedding.from_word2vec_format(
                    filepath=None,
                    binary=True),
                string_entities_formatter=StringEntitiesSimpleFormatter(),
                string_emb_entity_formatter=StringEntitiesSimpleFormatter())),
        frames_collection=RuSentiFramesCollection.read_collection(version=RuSentiFramesVersions.V20),
        frame_role_label_scaler=ThreeLabelScaler(),
        entity_to_group_func=entity_to_group_func,
        pos_terms_mapper=PosTermsMapper(None))

    ###########################
    # Step 3. Serialize data
    ###########################

    NetworkInputHelper.prepare(
        experiment=None,                 # Remove experiment
        terms_per_context=None,
        balance=None)

    ###########################
    # Step 4. Deserialize data
    ###########################

    network = PiecewiseCNN()
    config = CNNConfig()

    config.set_term_embedding(EmbeddingHelper.load_vocab("embedding.txt"))

    inference_ctx = InferenceContext.create_empty()
    inference_ctx.initialize(
        dtypes=[DataType.Test],
        create_samples_view_func=lambda data_type: BaseSampleStorageView(
            storage=BaseRowsStorage.from_tsv("samples.txt"),
            row_ids_provider=MultipleIDProvider()),
        has_model_predefined_state=True,
        vocab=EmbeddingHelper.load_vocab("vocab.txt"),
        labels_count=3,
        input_shapes=NetworkInputShapes(iter_pairs=[
            (NetworkInputShapes.FRAMES_PER_CONTEXT, config.FramesPerContext),
            (NetworkInputShapes.TERMS_PER_CONTEXT, config.TermsPerContext),
            (NetworkInputShapes.SYNONYMS_PER_CONTEXT, config.SynonymsPerContext),
        ]),
        bag_size=config.BagSize)

    ############################
    # Step 5. Model preparation.
    ############################

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

    model.predict()

    ########################################################
    # Step 6. Gather annotated contexts onto document level.
    ########################################################

    labeled_samples = model.get_labeled_samples_collection(data_type=DataType.Test)

    # TODO. For now it is limited to tsv.
    # TODO. For now it is limited to tsv.
    # TODO. For now it is limited to tsv.
    with TsvPredictProvider(filepath="out.txt") as out:
        out.load(sample_id_with_uint_labels_iter=labeled_samples.iter_non_duplicated_labeled_sample_row_ids(),
                 labels_scaler=labels_scaler)


if __name__ == '__main__':

    extract(EXAMPLES["simple"])
