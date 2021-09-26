from arekit.common.entities.formatters.str_simple_fmt import StringEntitiesSimpleFormatter
from arekit.common.experiment.annot.single_label import DefaultSingleLabelAnnotationAlgorithm
from arekit.common.experiment.data_type import DataType
from arekit.common.experiment.input.providers.columns.opinion import OpinionColumnsProvider
from arekit.common.experiment.input.providers.label.multiple import MultipleLabelProvider
from arekit.common.experiment.input.providers.opinions import OpinionProvider
from arekit.common.experiment.input.providers.row_ids.multiple import MultipleIDProvider
from arekit.common.experiment.input.providers.rows.opinions import BaseOpinionsRowProvider
from arekit.common.experiment.input.providers.text.single import BaseSingleTextProvider
from arekit.common.experiment.input.readers.tsv_sample import TsvInputSampleReader
from arekit.common.experiment.input.repositories.opinions import BaseInputOpinionsRepository
from arekit.common.experiment.input.repositories.sample import BaseInputSamplesRepository
from arekit.common.experiment.input.storages.tsv_opinion import TsvOpinionsStorage
from arekit.common.experiment.input.storages.tsv_sample import TsvSampleStorage
from arekit.common.news.parse_options import NewsParseOptions
from arekit.common.frame_variants.collection import FrameVariantsCollection
from arekit.common.labels.base import NoLabel
from arekit.common.news.base import News

from arekit.contrib.experiment_rusentrel.common import entity_to_group_func
from arekit.contrib.experiment_rusentrel.labels.scalers.three import ThreeLabelScaler
from arekit.contrib.experiment_rusentrel.synonyms.provider import RuSentRelSynonymsCollectionProvider
from arekit.contrib.networks.context.architectures.pcnn import PiecewiseCNN
from arekit.contrib.networks.context.configurations.cnn import CNNConfig
from arekit.contrib.networks.core.ctx_inference import InferenceContext
from arekit.contrib.networks.core.feeding.bags.collection.single import SingleBagsCollection
from arekit.contrib.networks.core.input.formatters.pos_mapper import PosTermsMapper
from arekit.contrib.networks.core.input.helper import NetworkInputHelper
from arekit.contrib.networks.core.input.helper_embedding import EmbeddingHelper
from arekit.contrib.networks.core.input.providers.columns import NetworkSampleColumnsProvider
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
from arekit.processing.text.parser import TextParser

from examples.input import EXAMPLES
from examples.network.embedding import RusvectoresEmbedding


def extract(text):

    ########################
    # Step 1. Parse text.
    ########################

    sentences = text  # TODO. split text onto sentences.
    stemmer = MystemWrapper()

    news = News(news_id=0,
                sentences=sentences)

    parse_options = NewsParseOptions(
        parse_entities=False,
        frame_variants_collection=FrameVariantsCollection(),
        stemmer=stemmer)

    parsed_news = TextParser.parse_news(news=news,
                                        parse_options=parse_options)

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

    # We pass it into NetworkInputEncoder
    opinion_provider = OpinionProvider.create(
        read_news_func=lambda _: parsed_news,
        iter_news_opins_for_extraction=lambda _: opins_for_extraction,
        parsed_news_it_func=lambda: [parsed_news],
        terms_per_context=50)

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

    # TODO. Use this instead.
    NetworkInputHelper.prepare(
        experiment=None,                 # Remove experiment
        terms_per_context=None,
        balance=None)

    # samples_repo = BaseInputSamplesRepository(
    #     columns_provider=NetworkSampleColumnsProvider(store_labels=True),
    #     rows_provider=sample_row_provider,
    #     storage=TsvSampleStorage(balance=False, write_header=True))

    # opinions_repo = BaseInputOpinionsRepository(
    #     columns_provider=OpinionColumnsProvider(),
    #     rows_provider=BaseOpinionsRowProvider(),
    #     storage=TsvOpinionsStorage())

    # samples_repo.populate(opinion_provider=opinion_provider,
    #                       target="samples.txt",
    #                       desc="sample")

    # opinions_repo.populate(opinion_provider=opinion_provider,
    #                        target="opinions.txt",
    #                        desc="opinion")

    # EmbeddingHelper.save_vocab(data=None, target="vocab.txt")
    # EmbeddingHelper.save_embedding(data=None, target="embedding.txt")

    ###########################
    # Step 4. Deserialize data
    ###########################

    network = PiecewiseCNN()
    config = CNNConfig()

    config.TermEmbeddingMatrix(EmbeddingHelper.load_vocab("embedding.txt"))

    inference_ctx = InferenceContext.create_empty()
    inference_ctx.initialize(
        dtypes=[DataType.Test],
        create_samples_reader_func=TsvInputSampleReader.from_tsv(
            filepath="samples.txt",
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